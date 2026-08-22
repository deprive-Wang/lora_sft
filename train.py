"""使用预编码的 SFT JSONL 训练 QLoRA adapter。

`dataset.py` 已经生成最终的 Qwen chat template token ids 和仅对 assistant
回复计算 loss 的 labels。本脚本只校验这些记录并在 batch 内动态 padding，
不会重复套用模板、分词或重建 label mask。

用法：
  python train.py --check-only
  python train.py
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from transformers import TrainingArguments

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
IGNORE_INDEX = -100
MAX_LEN = 1024

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_PATH = REPO_ROOT / "data" / "sft_train.jsonl"
DEFAULT_VAL_PATH = REPO_ROOT / "data" / "sft_val.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "checkpoints" / "qwen2_5_1_5b_ultrachat_lora"
DEFAULT_LOG_DIR = REPO_ROOT / "experiments" / "qwen2_5_1_5b_ultrachat_lora"

TokenizedSample: TypeAlias = dict[str, list[int]]
BatchLists: TypeAlias = dict[str, list[list[int]]]
TrainingArgumentsFactory: TypeAlias = Callable[..., "TrainingArguments"]


def validate_sample(sample: object, path: Path, line_number: int) -> TokenizedSample:
    """校验 `dataset.py` 产出的 token 级样本契约。"""
    if not isinstance(sample, dict):
        raise ValueError(f"{path}:{line_number}: expected a JSON object")

    input_ids = sample.get("input_ids")
    labels = sample.get("labels")
    if not isinstance(input_ids, list) or not isinstance(labels, list):
        raise ValueError(
            f"{path}:{line_number}: input_ids and labels must both be JSON arrays"
        )
    if not input_ids or len(input_ids) != len(labels):
        raise ValueError(
            f"{path}:{line_number}: input_ids and labels must be non-empty and equal length"
        )
    if len(input_ids) > MAX_LEN:
        raise ValueError(
            f"{path}:{line_number}: sequence length {len(input_ids)} exceeds {MAX_LEN}"
        )
    if any(type(token_id) is not int or token_id < 0 for token_id in input_ids):
        raise ValueError(f"{path}:{line_number}: input_ids must contain non-negative integers")
    if any(type(label) is not int for label in labels):
        raise ValueError(f"{path}:{line_number}: labels must contain integers")

    if any(
        label != IGNORE_INDEX and label != token_id
        for token_id, label in zip(input_ids, labels)
    ):
        raise ValueError(
            f"{path}:{line_number}: unmasked labels must equal their input_ids"
        )
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError(
            f"{path}:{line_number}: sample has no assistant tokens for loss"
        )

    return {"input_ids": input_ids, "labels": labels}


def load_tokenized_samples(path: Path) -> list[TokenizedSample]:
    """读取并校验预编码 JSONL，不执行二次分词。"""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python dataset.py` before starting training."
        )

    samples: list[TokenizedSample] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_sample = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            samples.append(validate_sample(raw_sample, path, line_number))

    if not samples:
        raise ValueError(f"{path}: split is empty")
    return samples


def pad_batch(features: Sequence[TokenizedSample], pad_token_id: int) -> BatchLists:
    """在 batch 内右侧 padding，并保留 assistant-only labels。"""
    if not features:
        raise ValueError("Cannot collate an empty batch")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("pad_token_id must be a non-negative integer")

    max_length = max(len(feature["input_ids"]) for feature in features)
    batch_input_ids: list[list[int]] = []
    batch_labels: list[list[int]] = []
    batch_attention_mask: list[list[int]] = []

    for feature in features:
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        padding_length = max_length - len(input_ids)
        batch_input_ids.append(input_ids + [pad_token_id] * padding_length)
        batch_labels.append(labels + [IGNORE_INDEX] * padding_length)
        batch_attention_mask.append([1] * len(input_ids) + [0] * padding_length)

    return {
        "input_ids": batch_input_ids,
        "labels": batch_labels,
        "attention_mask": batch_attention_mask,
    }


class TokenIdDataCollator:
    """将动态 padding 后的 token id batch 转为 PyTorch tensor。"""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[TokenizedSample]) -> dict[str, object]:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "PyTorch is required for training. Install the cloud training dependencies first."
            ) from error

        batch = pad_batch(features, self.pad_token_id)
        return {
            name: torch.tensor(values, dtype=torch.long)
            for name, values in batch.items()
        }


def check_padding(samples: Sequence[TokenizedSample]) -> None:
    """不加载 tokenizer 或模型，验证动态右侧 padding。"""
    first_sample = samples[0]
    second_sample = next(
        (
            sample
            for sample in samples[1:]
            if len(sample["input_ids"]) != len(first_sample["input_ids"])
        ),
        first_sample,
    )

    sample_batch = [first_sample, second_sample]
    batch = pad_batch(sample_batch, pad_token_id=0)
    lengths = {len(row) for row in batch["input_ids"]}
    if len(lengths) != 1:
        raise AssertionError("Padded input_ids have inconsistent lengths")

    for feature, input_ids, labels, attention_mask in zip(
        sample_batch,
        batch["input_ids"],
        batch["labels"],
        batch["attention_mask"],
    ):
        original_length = len(feature["input_ids"])
        if input_ids[:original_length] != feature["input_ids"]:
            raise AssertionError("Padding changed an input token")
        if labels[:original_length] != feature["labels"]:
            raise AssertionError("Padding changed a label")
        if any(label != IGNORE_INDEX for label in labels[original_length:]):
            raise AssertionError("Padding labels must use IGNORE_INDEX")
        expected_mask = [1] * original_length + [0] * (len(input_ids) - original_length)
        if attention_mask != expected_mask:
            raise AssertionError("attention_mask does not match right padding")


def load_training_components():
    """仅在退出 check-only 后导入 GPU 训练依赖。"""
    try:
        import accelerate
        import bitsandbytes
        import tensorboard
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise RuntimeError(
            "Training requires torch, transformers, peft, bitsandbytes, accelerate, and "
            "tensorboard in the GPU environment."
        ) from error

    return (
        torch,
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )


def build_training_arguments(
    training_arguments_class: TrainingArgumentsFactory,
    output_dir: Path,
    logging_dir: Path,
) -> "TrainingArguments":
    """构造固定训练参数，并兼容 Transformers 4.x 与 5.x。"""
    argument_names = inspect.signature(training_arguments_class).parameters
    evaluation_key = (
        "eval_strategy" if "eval_strategy" in argument_names else "evaluation_strategy"
    )
    arguments = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "learning_rate": 2e-4,
        "num_train_epochs": 2,
        "fp16": True,
        "bf16": False,
        "gradient_checkpointing": True,
        "logging_steps": 10,
        "logging_first_step": True,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "report_to": ["tensorboard"],
        "remove_unused_columns": False,
        "seed": 42,
        evaluation_key: "epoch",
    }
    if "logging_dir" in argument_names:
        arguments["logging_dir"] = str(logging_dir)
    return training_arguments_class(**arguments)


def run_training(
    args: argparse.Namespace,
    train_samples: list[TokenizedSample],
    val_samples: list[TokenizedSample],
) -> None:
    """在支持 CUDA 的 Linux GPU 环境运行 4-bit QLoRA 训练。"""
    (
        torch,
        lora_config_class,
        task_type,
        get_peft_model,
        prepare_model_for_kbit_training,
        auto_model_for_causal_lm,
        auto_tokenizer,
        bits_and_bytes_config,
        trainer_class,
        training_arguments_class,
    ) = load_training_components()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for 4-bit QLoRA training. Run this command in the 3090 GPU environment."
        )

    tokenizer = auto_tokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer has no usable pad_token_id")

    quantization_config = bits_and_bytes_config(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = auto_model_for_causal_lm.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        lora_config_class(
            task_type=task_type.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    # Transformers 5.x moved TensorBoard's directory out of TrainingArguments.
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(args.logging_dir)

    training_arguments = build_training_arguments(
        training_arguments_class, args.output_dir, args.logging_dir
    )
    trainer = trainer_class(
        model=model,
        args=training_arguments,
        train_dataset=train_samples,
        eval_dataset=val_samples,
        data_collator=TokenIdDataCollator(tokenizer.pad_token_id),
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    evaluation_metrics = trainer.evaluate()
    trainer.log_metrics("eval", evaluation_metrics)
    trainer.save_metrics("eval", evaluation_metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a QLoRA adapter from tokenized SFT JSONL")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--logging-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--resume-from-checkpoint", type=str)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate JSONL records and dynamic padding without loading a model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_samples = load_tokenized_samples(args.train_path)
    val_samples = load_tokenized_samples(args.val_path)
    check_padding(train_samples)
    check_padding(val_samples)
    print(f"Validated {len(train_samples)} train and {len(val_samples)} validation samples.")

    if args.check_only:
        print("Check-only completed without loading a tokenizer, model, or GPU dependencies.")
        return

    run_training(args, train_samples, val_samples)


if __name__ == "__main__":
    main()
