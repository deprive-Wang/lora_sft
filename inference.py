"""本地推理验证：4-bit 基座 + 训练好的 LoRA adapter 对话对比。

同一组 prompt 分别在 adapter 关闭（裸基座）与开启（微调后）两种状态下
生成，直观检查 QLoRA 微调对模型行为的影响。默认与训练时一致的 4-bit
NF4 加载；本地 bitsandbytes 不可用时可加 --no-4bit 退回 FP16 基座。

用法：
  python inference.py
  python inference.py --no-4bit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ADAPTER_PATH = REPO_ROOT / "checkpoints" / "qwen2_5_1_5b_ultrachat_lora"

TEST_PROMPTS = [
    "请用两三句话解释什么是过拟合。",
    "Write a short paragraph explaining why the sky is blue.",
    "我最近在学大模型微调，能给我一些学习建议吗？",
]


def build_model(model_source: str, load_in_4bit: bool):
    """加载基座，默认使用与训练一致的 4-bit NF4 配置。"""
    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        if load_in_4bit
        else None
    )
    return AutoModelForCausalLM.from_pretrained(
        model_source,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )


def generate_reply(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    """对单条 user prompt 生成回复，只解码新增 token。"""
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {
        name: tensor.to(model.device)
        for name, tensor in encoded.items()
        if name in ("input_ids", "attention_mask")
    }
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base and LoRA-tuned replies")
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local base model directory; defaults to the hub id Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Load the base model in FP16 instead of 4-bit NF4",
    )
    args = parser.parse_args()

    if not args.adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {args.adapter_path}")
    if args.model_path is not None and not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    model_source = str(args.model_path) if args.model_path else MODEL_ID

    if not torch.cuda.is_available():
        print("WARNING: CUDA unavailable, running on CPU will be very slow.")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    model = PeftModel.from_pretrained(
        build_model(model_source, not args.no_4bit), args.adapter_path
    )
    model.eval()

    for prompt in TEST_PROMPTS:
        print(f"\n{'=' * 60}\nPROMPT: {prompt}\n{'=' * 60}")
        with model.disable_adapter():
            print(f"---- BASE ----\n{generate_reply(model, tokenizer, prompt)}")
        print(f"---- SFT (LoRA) ----\n{generate_reply(model, tokenizer, prompt)}")


if __name__ == "__main__":
    main()
