"""本地对比 eval loss：adapter 关闭（基座）vs 开启（微调后）。

同一硬件、同一 4-bit 量化配置下对 data/sft_val.jsonl 各评估一遍，
两个数的差值即 QLoRA 微调在验证集上的净收益。loss 口径与 Trainer
一致：逐 batch 的 token 级均值 loss 再对 batch 求算术平均。

用法：
  python eval_baseline.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from train import TokenIdDataCollator, load_tokenized_samples

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ADAPTER_PATH = REPO_ROOT / "checkpoints" / "qwen2_5_1_5b_ultrachat_lora"
DEFAULT_VAL_PATH = REPO_ROOT / "data" / "sft_val.jsonl"


def build_model():
    """与训练一致的 4-bit NF4 基座加载，保证数值口径可比。"""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )


@torch.inference_mode()
def evaluate_loss(model, samples, collator, batch_size: int = 2) -> float:
    """逐 batch 前向取 loss，返回 batch 均值（Trainer 的评估口径）。"""
    losses = []
    for start in range(0, len(samples), batch_size):
        batch = collator(samples[start : start + batch_size])
        batch = {name: tensor.to(model.device) for name, tensor in batch.items()}
        losses.append(model(**batch).loss.item())
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare eval loss with adapter on/off")
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    args = parser.parse_args()

    if not args.adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {args.adapter_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("This comparison expects the local CUDA GPU.")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    if tokenizer.pad_token_id is None:
        raise ValueError("The saved tokenizer has no pad_token_id")
    samples = load_tokenized_samples(args.val_path)
    collator = TokenIdDataCollator(tokenizer.pad_token_id)

    model = PeftModel.from_pretrained(build_model(), args.adapter_path)
    model.eval()

    with model.disable_adapter():
        base_loss = evaluate_loss(model, samples, collator)
    sft_loss = evaluate_loss(model, samples, collator)

    print(f"val samples: {len(samples)}")
    print(f"BASE (adapter off) eval loss: {base_loss:.4f}")
    print(f"SFT  (adapter on)  eval loss: {sft_loss:.4f}")
    print(f"delta: {base_loss - sft_loss:+.4f}")


if __name__ == "__main__":
    main()
