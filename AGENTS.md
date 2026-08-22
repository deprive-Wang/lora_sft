# LoRA/QLoRA SFT 学习项目

在 Qwen2.5-1.5B-Instruct 上做小规模指令微调，核心是 label mask、LoRA 与 4-bit QLoRA 训练流程。工具链：transformers + peft + bitsandbytes + datasets。

## 目录与边界

| 路径 | 职责 |
|---|---|
| `dataset.py` | chat template 编码、label mask（两次完整编码 + 前缀校验）、自检与 train/val 落盘 |
| `train.py` | 校验预编码 JSONL、动态右 padding、加载 NF4 量化基座并训练 LoRA adapter |
| `inference.py` | 本地推理对比：4-bit 基座 + adapter，同一 prompt 在 adapter 开/关两种状态下生成 |
| `eval_baseline.py` | 本地 eval loss 对比：adapter 开/关在同一验证集上的 loss 差值 |
| `scripts/prepare_data.py` | 从 HuggingFace streaming 下载 ultrachat_200k，抽取首轮单轮样本 |
| `data/` | 原始 JSONL 与 token id 级训练文件（gitignore） |
| `checkpoints/` | adapter 与训练状态（gitignore） |
| `experiments/` | TensorBoard 日志与实验产物（gitignore） |
| `tf-logs/` | 从云端拉回的 TensorBoard 事件文件（gitignore） |
| `实验报告.md` | 用户复习用的实验总结（背景/原理/结果/踩坑）；实验结论变化时应同步更新 |
| `thesis/` | LoRA/QLoRA 阅读资料；除非用户要求，不要改动或移动 |

不要在 `train.py` 中重新套 chat template、重新分词或重建 labels；训练输入必须来自 `dataset.py` 生成的 token id JSONL。

## 运行环境

- 本地 Python 使用 conda 环境 `lora_sft`（Miniconda），解释器为 `E:\miniconda\envs\lora_sft\python.exe`；该环境已安装 PyTorch。
- 本地数据准备、`dataset.py` 与 `python train.py --check-only` 都在此环境执行。不要用 Anaconda `base`，也不要新建同名环境。
- README 里的 `LoRA_SFT` 与实际环境名不一致，以 `lora_sft` 为准。
- 本地 GPU 为 RTX 3070 Laptop 8GB：4-bit 推理与 eval 已验证可在 Windows 本地跑（bitsandbytes 0.50.1 + torch 2.5.1+cu121），但不做完整训练。
- 完整 `python train.py`（4-bit QLoRA）仍要求 Linux CUDA + bitsandbytes（目标为 AutoDL 3090），不要默认在 Windows 上直接训练。
- 本机 `HF_HOME=D:\AI_model\huggingface`（系统环境变量，缓存不在 C 盘默认位置）；基座 Qwen2.5-1.5B-Instruct（3.09 GB）已缓存。本地加载模型加 `HF_HUB_OFFLINE=1` 避免联网校验。

## 常用命令

先激活本地环境：

```powershell
conda activate lora_sft
```

然后执行：

```bash
python scripts/prepare_data.py        # 下载并抽取单轮样本
python dataset.py --check-only        # 编码自检与长度统计，不需 GPU
python dataset.py                     # 生成 data/sft_train.jsonl / data/sft_val.jsonl
python train.py --check-only          # 校验训练 JSONL 与动态 padding，不加载模型
python train.py                       # Linux CUDA GPU 上运行 QLoRA 训练
HF_HUB_OFFLINE=1 python inference.py     # 本地 4-bit 推理对比（--model-path 可指定本地模型目录）
HF_HUB_OFFLINE=1 python eval_baseline.py # 本地 eval loss 对比（基座 vs 微调）
```

项目没有独立的 pytest、lint、typecheck 或 build 配置；修改数据/训练逻辑后至少运行对应的 `--check-only` 命令。HF 网络分场景：云端（AutoDL）用 `HF_ENDPOINT=https://hf-mirror.com`；**本地不要用 hf-mirror**（对 LFS 权重 308 重定向回官方站，transformers 5 会解析失败），本机直连可用，且下载必须加 `HF_HUB_DISABLE_XET=1`（默认 xet 后端在本机反复停滞）。

## 关键设计决策

- **label mask**：`dataset.py` 用两次完整 `apply_chat_template` 编码 + 前缀校验保证 mask 边界与 token 边界对齐；不手动拼字符串、不逐条单独编码（BPE 跨边界合并问题）。
- **IGNORE_INDEX = -100**：CrossEntropyLoss 的 `ignore_index`，被 mask 位置不产生梯度。
- **max_len = 1024**：超长样本直接丢弃，不截断。
- **数据划分**：目标为 9,500 train / 500 val；有效样本不足时按同比例自动切分。
- **QLoRA 固定参数**：4-bit NF4 + double quantization；LoRA rank 16、alpha 32、dropout 0.05，目标模块为 `q_proj`/`k_proj`/`v_proj`/`o_proj`；训练配置为 micro batch 2、梯度累积 16、learning rate `2e-4`、FP16、2 epochs。
- **训练平台**：完整 QLoRA 入口要求 CUDA 与 bitsandbytes，脚本目标环境为 Linux GPU；本地无 GPU 时只运行 `--check-only`。
- **transformers 4.x/5.x 兼容**：`TrainingArguments` 的参数按构造签名探测传入（`eval_strategy`/`evaluation_strategy`、`logging_dir`）；5.x 的 TensorBoard 目录通过 `TENSORBOARD_LOGGING_DIR` 环境变量指定（`train.py` 的 `run_training` 中设置）。`torch_dtype` 在 5.x 仅为 deprecation 警告、仍可用，为兼容 4.x 不改名。
- **训练已完成（2026-08-22）**：594 步 / 67 分钟，最终 adapter 在 `checkpoints/qwen2_5_1_5b_ultrachat_lora/`，指标见同目录 `trainer_state.json`；baseline 对比与推理验证结论见 `实验报告.md`。

## 编码规范

- Python 3.12，PEP 8，4 空格缩进。
- 文件路径使用 `pathlib.Path`。
- 公共函数、方法和公开类属性提供类型注解。
- 注释只说明设计动机、边界取舍或非显然约束，不复述代码。
- `dataset.py` 保持不依赖 torch，使编码和数据自检可在无 GPU 环境运行。

## 外部依赖与稳定性

- `requirements.txt` 区分本地数据准备依赖与云端训练依赖；torch 使用训练镜像自带版本，不要随意覆盖。
- 基座模型和 tokenizer 为 `Qwen/Qwen2.5-1.5B-Instruct`，首次运行会从 HuggingFace 下载 tokenizer。
- `data/`、`checkpoints/`、`experiments/`、`tf-logs/` 不入库；不要提交数据集、模型权重或实验输出。
- `inference.py` / `eval_baseline.py` 复用 `train.py` 的数据加载与 collator（`TokenIdDataCollator`、`load_tokenized_samples`），修改这些契约时三个脚本需保持一致。
