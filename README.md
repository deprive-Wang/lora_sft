# LoRA/QLoRA 指令微调

在 Qwen2.5-1.5B-Instruct 上完成小规模 SFT：用 4-bit NF4 量化加载基座，以 LoRA 适配器在 1 万条单轮指令数据上训练，对比基座 / 加载 adapter / 合并 adapter 三种推理形态的输出。模型本体不再手写——本项目的学习重点是预训练模型、chat template、label mask、量化与参数高效微调的真实工程流程，工具链为 transformers + peft + bitsandbytes。对照资料是 LoRA / QLoRA 论文与 Hugging Face PEFT 文档，只对照、不照抄。

本文档同时是项目说明和实验记录（实验部分待训练完成后填写，当前不含任何未实际发生的结果）。

## 固定规格

| 项 | 值 |
|---|---|
| 基座 | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| 数据 | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) train_sft 每条对话的首轮交换（单轮化） |
| 划分 | 9,500 train / 500 validation，最大长度 1,024 tokens，超长丢弃 |
| 量化 | 4-bit NF4 + double quantization（bitsandbytes） |
| LoRA | rank 16，alpha 32，dropout 0.05，目标 q_proj / k_proj / v_proj / o_proj |
| 训练 | micro batch 2 × 梯度累积 16 = 有效 batch 32，lr 2e-4，FP16，2 epochs |
| 预计步数 | 9,500 × 2 / 32 ≈ 594 optimizer steps |

## label mask

SFT 的 loss 只对 assistant 回复段计算。实现采用两次完整 chat template 编码加前缀校验：

```text
prompt_ids = chat_template([user], add_generation_prompt=True)   # 条件段
full_ids   = chat_template([user, assistant])                   # 条件段 + 回复
labels     = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
```

-100 是 CrossEntropyLoss 的 ignore_index。不手动拼字符串、不逐条 message 单独编码，因为 BPE 跨边界合并会让分段编码与整段编码不一致；前缀校验保证 mask 边界与 token 边界对齐。实现与自检见 `dataset.py`。

## 验收标准

```text
1. 训练前记录基座模型在固定 prompt 上的输出，并核对可训练参数占比。
2. 保存 adapter、tokenizer、训练配置、loss 和 validation 指标。
3. 用同一组 prompt 比较基座、加载 adapter、合并 adapter 三种推理结果。
4. 记录训练耗时、峰值显存、典型改进样例与失败样例。
5. 明确说明微调是任务/风格适配，不能把结果误解为可靠的新事实注入。
```

## 快速开始

### 本地（数据准备与自检，无需 GPU）

```powershell
conda activate LoRA_SFT
pip install -r requirements.txt   # 本地只需要 transformers + datasets 两节

python scripts/prepare_data.py    # 下载并抽取单轮样本 -> data/ultrachat_single_turn_raw.jsonl
python dataset.py --check-only    # 编码自检 + 长度统计
python dataset.py                 # 生成 data/sft_train.jsonl / sft_val.jsonl（token id 级）
```

首次运行会自动下载 Qwen2.5 tokenizer（约 11MB）。国内网络受限时先设镜像：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

### 云端训练（AutoDL 3090）

待 `train.py` 完成后补充。

## 目录结构

```text
lora_sft/
  scripts/prepare_data.py   ultrachat_200k 流式抽取首轮交换 -> 原始 jsonl
  dataset.py                chat template 编码、label mask、自检与 train/val 落盘
  requirements.txt          本地与云端依赖（torch 由镜像自带，不列出）
  data/                     gitignore；原始 jsonl 与 token id 级 train/val jsonl
  checkpoints/              gitignore；adapter 与训练状态
  experiments/              gitignore；日志、曲线与推理对比样例
```

## 实验记录

待训练完成后填写：基座固定 prompt 输出、可训练参数占比、loss/validation 曲线、训练耗时、峰值显存、三态推理对比样例（含失败样例）、结论与下一版改进。

## 参考

- LoRA: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- QLoRA: [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- [Hugging Face PEFT 文档](https://huggingface.co/docs/peft)
