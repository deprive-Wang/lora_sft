# LoRA/QLoRA SFT 学习项目

在 Qwen2.5-1.5B-Instruct 上做小规模指令微调，核心是 label mask 与量化训练流程。
工具链：transformers + peft + bitsandbytes + datasets。

## 目录结构

| 路径 | 职责 |
|---|---|
| `dataset.py` | chat template 编码、label mask（两次完整编码 + 前缀校验）、自检与 train/val 落盘 |
| `scripts/prepare_data.py` | 从 HuggingFace streaming 下载 ultrachat_200k，抽取首轮单轮样本 |
| `data/` | 原始 jsonl 与 token id 级训练文件（gitignore） |
| `requirements.txt` | 本地（数据准备）与云端（训练）依赖 |

## 常用命令

```bash
# 数据准备
python scripts/prepare_data.py        # 抽取单轮样本 -> data/ultrachat_single_turn_raw.jsonl
python dataset.py --check-only          # 编码自检 + 长度统计（不需 GPU）
python dataset.py                       # 生成 data/sft_train.jsonl / sft_val.jsonl

# 云端训练（train.py 待完成）
# python train.py
```

国内 HF 镜像：`export HF_ENDPOINT=https://hf-mirror.com`

## 关键设计决策

- **label mask**：`dataset.py` 用两次完整 `apply_chat_template` 编码 + 前缀校验保证 mask 边界与 token 边界对齐；不手动拼字符串、不逐条单独编码（BPE 跨边界合并问题）。
- **IGNORE_INDEX = -100**：CrossEntropyLoss 的 ignore_index，被 mask 位置不产生梯度。
- **max_len = 1024**：超长样本直接丢弃，不截断。
- **数据划分**：9,500 train / 500 val，有效样本不足时按同比例自动切分。

## 编码规范

- Python 3.12，PEP 8，4 空格缩进。
- 文件路径用 `pathlib.Path`。
- 公共函数有类型注解。
- 注释说明设计动机，不复述代码逻辑。
- `dataset.py` 不依赖 torch，纯 Python 可在本地无 GPU 环境运行。

## 已知约束

- `train.py` 尚未完成，训练流程仅靠 README 记录了计划参数。
- 基座模型 Qwen2.5-1.5B-Instruct 约 11MB tokenizer 文件首次自动下载。
- `data/`、`checkpoints/`、`experiments/` 目录 gitignore，不入库。
