"""把单轮对话编码成 Qwen2.5 的 SFT 训练样本，核心是 label mask。

一条样本的 loss 只应来自 assistant 回复段：SFT 的学习目标是
"给定指令，生成回复"；如果把 prompt 段也算进 loss，相当于额外要求模型
背诵用户指令，既浪费训练信号，也让指标失真。

实现采用"两次完整模板编码 + 前缀校验"：

  prompt_ids = chat_template([user], add_generation_prompt=True)
               编码到 "<|im_start|>assistant\\n" 为止 —— 这正好是推理时
               模型真实可见的前缀（生成 prompt 的定义）
  full_ids   = chat_template([user, assistant])
               在前缀之后接上回复正文与 <|im_end|>
  labels     = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

  -100 是 CrossEntropyLoss(ignore_index=-100) 的约定，被 mask 的位置不产生梯度。

  为什么不手动拼字符串、也不逐条 message 单独编码：BPE 存在跨边界合并，
  分段编码再拼接与整段编码的 token 序列不一定相同。两次完整编码后校验
  prompt_ids 是 full_ids 的严格前缀，mask 边界才与 token 边界对齐。

本文件不依赖 torch：编码、校验、统计、落盘全部用纯 Python 完成，
数据在本地（无 GPU）验证好之后再上传云端训练，云端无需再访问
HuggingFace 网络下载数据集。

用法：
  python dataset.py               # 编码 + 自检 + 生成 data/sft_train.jsonl / sft_val.jsonl
  python dataset.py --check-only  # 只做编码自检与长度统计，不落盘
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # 只需下载 tokenizer 文件（约 11MB）
IGNORE_INDEX = -100
MAX_LEN = 1024
N_TRAIN = 9_500
N_VAL = 500

REPO_ROOT = Path(__file__).resolve().parent
RAW_PATH = REPO_ROOT / "data" / "ultrachat_single_turn_raw.jsonl"
TRAIN_PATH = REPO_ROOT / "data" / "sft_train.jsonl"
VAL_PATH = REPO_ROOT / "data" / "sft_val.jsonl"


def load_raw_samples(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}，请先运行 python scripts/prepare_data.py 生成原始数据"
        )
    samples = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def template_ids(
    tokenizer, messages: list[dict], add_generation_prompt: bool = False
) -> list[int]:
    """取 chat template 编码后的 token id 列表。

    transformers v4 的 apply_chat_template(tokenize=True) 直接返回 List[int]；
    v5 起改为返回 BatchEncoding（含 input_ids / attention_mask 两个 key），
    len() 是 key 数量、整体比较是字典对象比较，都不是 token 语义。
    这里统一收敛成平铺的 list[int]，两个版本行为一致。
    """
    out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    if isinstance(out, list) and out and isinstance(out[0], int):
        return out
    return list(out["input_ids"])


def encode_sample(
    messages: list[dict], tokenizer, max_len: int = MAX_LEN
) -> tuple[list[int], list[int]] | None:
    """编码一条单轮样本，返回 (input_ids, labels)；超过 max_len 返回 None。

    每条样本都执行前缀校验：一旦模板行为与预期不符立即抛错，
    而不是静默生成边界错误的 label。
    """
    prompt_ids = template_ids(tokenizer, messages[:1], add_generation_prompt=True)
    full_ids = template_ids(tokenizer, messages)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "chat template 前缀不一致：prompt 编码不是完整编码的前缀，"
            "mask 边界无法对齐，需要人工检查模板输出"
        )
    if len(full_ids) > max_len:
        return None
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(full_ids[len(prompt_ids):])
    return list(full_ids), labels


def show_sample(input_ids: list[int], labels: list[int], tokenizer) -> None:
    """打印一条样本的 mask 边界，供人工核对 loss 段确实只是 assistant 回复。"""
    if len(input_ids) != len(labels):
        raise AssertionError("input_ids 与 labels 长度不一致")
    masked = [token for token, label in zip(input_ids, labels) if label == IGNORE_INDEX]
    loss_tokens = [token for token, label in zip(input_ids, labels) if label != IGNORE_INDEX]
    if not loss_tokens:
        raise AssertionError("loss 段为空，整条样本都不会产生梯度")
    print(f"  总长 {len(input_ids)} tokens = 条件段 {len(masked)} + loss 段 {len(loss_tokens)}")
    print("  ---- 条件段末尾（应止于 assistant 生成提示）----")
    print(f"  {tokenizer.decode(masked[-24:])!r}")
    print("  ---- loss 段开头（应为 assistant 回复正文）----")
    text = tokenizer.decode(loss_tokens)
    print(f"  {text[:300]}{'...' if len(text) > 300 else ''}")


def percentile(sorted_lengths: list[int], q: float) -> int:
    return sorted_lengths[min(int(q * len(sorted_lengths)), len(sorted_lengths) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="编码 SFT 数据并核对 label mask")
    parser.add_argument(
        "--check-only", action="store_true", help="只自检与统计，不生成训练文件"
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    samples = load_raw_samples(RAW_PATH)
    print(f"读取 {len(samples)} 条原始样本，tokenizer: {MODEL_ID}")

    encoded: list[tuple[list[int], list[int]]] = []
    too_long = 0
    for sample in samples:
        result = encode_sample(sample["messages"], tokenizer)
        if result is None:
            too_long += 1
        else:
            encoded.append(result)

    lengths = sorted(len(ids) for ids, _ in encoded)
    print(f"编码成功 {len(encoded)} 条，超过 {MAX_LEN} tokens 丢弃 {too_long} 条")
    print(
        f"token 长度 p50={percentile(lengths, 0.50)} "
        f"p90={percentile(lengths, 0.90)} "
        f"p95={percentile(lengths, 0.95)} max={lengths[-1]}"
    )

    print("\n前 3 条样本的 mask 边界核对：")
    for ids, labels in encoded[:3]:
        show_sample(ids, labels, tokenizer)
        print()

    if args.check_only:
        print("--check-only：未生成训练文件")
        return

    if len(encoded) < N_TRAIN + N_VAL:
        n_val = max(1, len(encoded) * N_VAL // (N_TRAIN + N_VAL))
        n_train = len(encoded) - n_val
        print(f"有效样本不足 {N_TRAIN + N_VAL}，按同比例切分 {n_train} train / {n_val} val")
    else:
        n_train, n_val = N_TRAIN, N_VAL

    for path, split in (
        (TRAIN_PATH, encoded[:n_train]),
        (VAL_PATH, encoded[n_train : n_train + n_val]),
    ):
        with path.open("w", encoding="utf-8") as file:
            for ids, labels in split:
                file.write(json.dumps({"input_ids": ids, "labels": labels}) + "\n")
        print(f"写入 {len(split)} 条 -> {path}")


if __name__ == "__main__":
    main()
