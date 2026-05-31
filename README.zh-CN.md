# Fail2Fix-RL

**Teacher-guided Failure-to-Fix 强化学习框架。**

Fail2Fix-RL 是一个面向小模型数学推理的轻量级研究框架，用来训练模型从自己的失败 rollout 中学习纠错。当前实现围绕 GSM8K 风格可验证数学题、Qwen2.5-0.5B-Instruct 级别模型和单卡训练环境构建。

项目受到 CIPO 等 correction-oriented RLVR 方法启发，但仓库代码按“可复现框架”组织，不包含临时实验脚本、阶段性日志或结果堆料。

## 框架图

![Teacher-guided F2F framework](assets/f2f_framework.png)

整体分成两个阶段：

1. **离线教师纠偏**：先让学生模型采样 rollout，筛选失败或低通过率解答，再调用强教师模型诊断并修正，最后只保留通过 verifier 的纠错样本。
2. **在线 F2F 训练**：同一个 RL step 中同时生成普通解题 rollout 和纠错 rollout，用可验证奖励、纠错 replay、动态 anchor 比例和风险规避项更新策略模型。

## 为什么做这个

传统 RLVR 在数学题上通常只有二值奖励：最终答案对就是 1，错就是 0。这个信号客观，但会浪费失败轨迹里的细节。一个只差最后一步计算的 near-miss、一条思路基本正确但格式不合规的答案、以及完全跑偏的推理，在普通二值奖励下都可能是 0。

Fail2Fix-RL 把模型自己生成的失败答案重新变成训练材料。学生模型先正常解题，然后把部分候选答案作为“可能有错”的参考轨迹重新输入，让模型学习检查、保留或修正。

## 核心方法

每个在线 RL step 有两条流：

```text
题目
  -> 普通 rollout
  -> verifier
  -> 组内 RL advantage

题目 + 候选解答
  -> 纠错 rollout
  -> verifier
  -> F2F 纠错奖励 + 风险规避 shaping
```

关键机制：

- **Failure-to-fix replay**：纠错 prompt 来自当前策略自己生成的 rollout。
- **教师初始化**：用强教师模型生成 verified corrections，先对小模型做 correction SFT。
- **中等难度优先**：优先选择同组有对有错的问题，因为这类样本更容易产生有效梯度。
- **动态 rho 控制**：根据 retention 调整 correct anchor 与 failed anchor 的比例。
- **风险规避奖励**：如果模型把原本正确的解答改错，会受到额外惩罚。
- **确定性 verifier**：抽取最终答案，并用数值/符号等价规则做自动打分。

## 仓库范围

当前仓库只保留可复用框架代码：数据准备、学生采样、教师纠错数据构造、纠错 SFT、在线 F2F RL、GRPO baseline 和评测。checkpoint、原始数据、TensorBoard 日志、一次性监控脚本、分析脚本和结果表先不上传，后续可以按论文或 PPT 需要再精选。

## 目录结构

```text
assets/
  f2f_framework.png                 方法框架图。

remote_scripts/
  prepare_gsm8k_grpo_data.py        将 ModelScope GSM8K 转成 RL JSONL。
  collect_student_rollouts.py       采样多条 student rollout。
  build_teacher_corrections.py      构造 verified teacher correction SFT 数据。
  train_correction_sft.py           全参数 correction SFT。
  train_f2f_online_rl.py            在线 F2F RL 主训练脚本。
  train_grpo_base.py                Vanilla GRPO baseline。
  eval_gsm8k_subset.py              GSM8K 子集评测。
  eval_correction_sft.py            纠错 prompt 评测。

verifier_math.py                    数学答案抽取与等价比较。
requirements.txt                    Python 依赖。
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

采样学生模型 rollout：

```bash
python remote_scripts/collect_student_rollouts.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data data/gsm8k_grpo/train.jsonl \
  --output data/teacher_correction/student_rollouts_train.jsonl \
  --limit 512 \
  --group-size 8
```

构造教师纠错数据：

```bash
cp .env.example .env.teacher

python remote_scripts/build_teacher_corrections.py \
  --rollouts data/teacher_correction/student_rollouts_train.jsonl \
  --output data/teacher_correction/correction_sft_train.jsonl \
  --cache data/teacher_correction/teacher_cache.jsonl
```

运行 correction SFT：

```bash
python remote_scripts/train_correction_sft.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data data/teacher_correction/correction_sft_train.jsonl \
  --output-dir checkpoints/f2f_correction_sft
```

继续在线 F2F RL：

```bash
python remote_scripts/train_f2f_online_rl.py \
  --model checkpoints/f2f_correction_sft \
  --train-data data/gsm8k_grpo/train.jsonl \
  --eval-data data/gsm8k_grpo/test.jsonl \
  --output-dir checkpoints/f2f_online \
  --max-steps 500 \
  --batch-size 8 \
  --group-size 8 \
  --max-new-tokens 1024 \
  --tensorboard
```

运行 GRPO baseline：

```bash
python remote_scripts/train_grpo_base.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data data/gsm8k_grpo/train.jsonl \
  --output-dir checkpoints/grpo_base \
  --report-to-tensorboard
```

评测 checkpoint：

```bash
python remote_scripts/eval_gsm8k_subset.py \
  --model checkpoints/f2f_online/best_eval_checkpoint \
  --data data/gsm8k_grpo/test.jsonl \
  --limit 200 \
  --output-dir reports/eval_gsm8k
```

## 复现说明

- 训练脚本默认面向 CUDA 环境。
- Qwen2.5-0.5B 全参数实验在 48GB RTX 4090 环境下开发。
- API key 只应放在本地 `.env.teacher` 这类环境文件里。
- 原始数据、checkpoint、rollout、日志和 TensorBoard event 文件默认不进 git。

## 引用与定位

推荐表述：

```text
Fail2Fix-RL: Teacher-guided Failure-to-Fix Reinforcement Learning.
Research prototype, 2026.
```

相关论文：

```text
Correction Intention Preference Optimization
arXiv:2605.14539
https://arxiv.org/abs/2605.14539
```
