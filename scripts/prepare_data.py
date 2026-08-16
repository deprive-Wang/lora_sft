"""下载 ultrachat_200k 并抽取单轮指令样本。

数据集：HuggingFaceH4/ultrachat_200k —— UltraChat 200k 对话的清洗版。
它是多轮对话数据；本项目按暑期计划只取每条对话的第一轮
user/assistant 交换，构成单轮指令样本，后续由 dataset.py 编码并做 label mask。

为什么用 streaming：全量数据集 parquet 约数百 MB，而本项目只需要前 1.2 万条
对话的首轮交换；streaming 模式只顺序拉取用到的分片，不把整个数据集落盘。

国内网络访问 huggingface.co 受限时，先设置 HF 镜像再运行本脚本：
  PowerShell:  $env:HF_ENDPOINT = "https://hf-mirror.com"
  Bash:        export HF_ENDPOINT=https://hf-mirror.com

用法：
  python scripts/prepare_data.py

输出：
  data/ultrachat_single_turn_raw.jsonl
    每行一个 JSON 对象：
    {"messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}
"""

from __future__ import annotations

import json
import sys
from itertools import islice
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "data" / "ultrachat_single_turn_raw.jsonl"

# 最终计划保留 9,500 train + 500 validation = 10,000 条。
# dataset.py 编码时会丢弃超过 max_len=1024 的样本，这里多抽一些作为余量。
MAX_DIALOGS = 12_000


def extract_single_turn(dialog: dict) -> dict | None:
    """取一条对话的首轮 user/assistant 交换；格式不符时返回 None。

    外部数据是系统边界：字段缺失、role 顺序异常、内容为空都在这里拦下，
    只把结构完整的样本写盘，避免训练阶段才发现数据问题。
    """
    messages = dialog.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    user, assistant = messages[0], messages[1]
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        return None
    if not user.get("content") or not assistant.get("content"):
        return None
    return {
        "messages": [
            {"role": "user", "content": user["content"]},
            {"role": "assistant", "content": assistant["content"]},
        ]
    }


def main() -> None:
    samples: list[dict] = []
    skipped = 0
    try:
        # 流式读取：下载发生在迭代时，而不是 load_dataset 调用时
        dialogs = load_dataset(
            "HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True
        )
        for dialog in islice(dialogs, MAX_DIALOGS):
            sample = extract_single_turn(dialog)
            if sample is None:
                skipped += 1
            else:
                samples.append(sample)
    except Exception as error:
        print(f"下载或读取数据失败: {error}", file=sys.stderr)
        print(
            "如果卡在网络连接，设置 HF 镜像后重跑：\n"
            '  PowerShell:  $env:HF_ENDPOINT = "https://hf-mirror.com"\n'
            "  Bash:        export HF_ENDPOINT=https://hf-mirror.com",
            file=sys.stderr,
        )
        sys.exit(1)

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAW_PATH.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"写入 {len(samples)} 条 -> {RAW_PATH}")
    print(f"跳过格式不符 {skipped} 条（读取对话上限 {MAX_DIALOGS}）")


if __name__ == "__main__":
    main()
