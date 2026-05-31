# Fail2Fix-RL

**从失败推理轨迹中学习自我修正。**

Fail2Fix-RL 是一个面向小模型数学推理的轻量级研究框架。它围绕 GSM8K、Qwen2.5-0.5B-Instruct 和单卡训练环境构建，用来验证：小模型能否利用自己的失败 rollout 学到更稳定的纠错能力。

本项目受到 CIPO 等 correction-oriented RLVR 方法启发，但定位不是宣称完整复现论文，而是一个可运行、可监控、可分析的工程化实验框架。

## 核心想法

传统 RLVR 通常只给二值奖励：答案对就是 1，答案错就是 0。这个信号客观，但会丢掉失败轨迹里的细节。一条只差最后一步计算的 near-miss，和一条完全跑偏的推理，在普通二值奖励下都是 0。

Fail2Fix-RL 把模型自己生成的解答重新变成训练材料。模型先正常解题，然后把部分候选解答作为“可能有错”的参考轨迹重新输入模型，让模型学习检查、保留或修正。

## 方法结构

每个在线 RL step 包含两条数据流：

```text
原始题目
  -> 普通 rollout
  -> 数学答案验证器
  -> GRPO 风格组内 advantage

候选解答
  -> correction prompt
  -> 修正 rollout
  -> 同一个答案验证器
  -> correction objective + 风险规避奖励
```

关键机制：

- **在线 correction replay**：纠错样本来自当前策略的实时 rollout。
- **中等难度优先**：优先选择有对有错的题目，避免全对/全错样本提供低效梯度。
- **动态 rho 控制**：根据 retention 调整 correct anchor 与 failed anchor 的比例。
- **风险规避奖励**：如果模型把原本正确的解答改错，会受到额外惩罚。
- **教师纠错 SFT**：可先用强教师模型生成高质量纠错数据，对小模型做纠错能力初始化。
- **可验证奖励**：使用最终答案抽取和数值/符号等价比较做自动打分。
- **单卡监控**：提供 TensorBoard、OOM 检测和恢复辅助脚本。

## 当前实验流程

1. 准备 GSM8K train/test JSONL。
2. 用 Qwen2.5-0.5B-Instruct 采样 student rollouts。
3. 调用强教师模型构造经过验证的 correction SFT 数据。
4. 对小模型做全参数 correction SFT。
5. 从 correction SFT checkpoint 继续跑 online CIPO/Fail2Fix-RL。
6. 在固定 GSM8K 子集上评测 checkpoint。

## 当前仓库内容

首版仓库只包含训练框架代码、数据准备脚本、评测脚本和监控工具。实验结果表、图、checkpoint、原始数据和长日志先不放进仓库，后续可以单独挑选后再加入。

当前代码支持四类实验：

- Base 模型在 GSM8K 风格可验证数学数据上的评测。
- Vanilla GRPO 作为 RLVR baseline。
- Plain F2F/CIPO-style correction replay 训练。
- Teacher-guided F2F：先用 verified teacher corrections 初始化小模型，再进行在线自我纠偏 RL。

## 目录结构

```text
remote_scripts/
  prepare_gsm8k_grpo_data.py        准备 GSM8K RL JSONL。
  collect_student_rollouts.py       采样 student rollout。
  build_teacher_corrections.py      构造教师纠错 SFT 数据。
  train_correction_sft.py           全参数 correction SFT。
  train_cipo_online_grpo.py         Online Fail2Fix/CIPO-style RL 主训练脚本。
  train_dapo_grpo_base.py           Vanilla GRPO baseline。
  eval_gsm8k_subset.py              GSM8K 子集评测。
  eval_correction_sft.py            纠错 prompt 评测。
  monitor_final3676_cipo.py         长训练 OOM/恢复监控。
  analyze_gsm8k_rollout_errors.py   rollout 错误分析。

reports/
  fail2fix_summary/                 早期实验图表和表格。

experiment_notes/
  teacher_correction_stage2.md      第二阶段教师纠错实验日志。

verifier_math.py                    数学答案抽取与等价比较。
```

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

准备 GSM8K：

```bash
python remote_scripts/prepare_gsm8k_grpo_data.py \
  --output-dir data/gsm8k_grpo
```

运行一个小规模 smoke 实验：

```bash
python remote_scripts/train_cipo_online_grpo.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --train-data data/gsm8k_grpo/train.jsonl \
  --eval-data data/gsm8k_grpo/test.jsonl \
  --output-dir checkpoints/fail2fix_smoke \
  --train-limit 64 \
  --eval-limit 32 \
  --max-steps 10 \
  --batch-size 4 \
  --group-size 4 \
  --generation-batch-size 2 \
  --forward-batch-size 2 \
  --max-new-tokens 512 \
  --eval-max-new-tokens 512 \
  --tensorboard
```

如果需要教师模型 API，复制 `.env.example`，把真实密钥放在本地环境文件里，不要提交到 GitHub：

```bash
cp .env.example .env.teacher
```

## 复现实验注意事项

- 训练脚本默认面向 CUDA 环境。
- 当前全参数训练在 48GB RTX 4090 环境下开发。
- 权重、原始数据、日志和 TensorBoard event 文件不应提交到 GitHub。
- 当前指标默认使用 strict verifier。
- 实验仍在推进，精确阶段记录以 `experiment_notes/` 为准。

## 引用与定位

推荐表述：

> Fail2Fix-RL 是一个面向小模型数学推理的自我修正强化学习原型。它通过在线 correction replay、动态 anchor 比例和风险规避奖励，让模型从自身失败轨迹中学习修正能力。

相关论文：

```text
Correction Intention Preference Optimization
arXiv:2605.14539
https://arxiv.org/abs/2605.14539
```
