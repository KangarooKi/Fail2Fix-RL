# Fail2Fix-RL

**Teacher-guided Failure-to-Fix 强化学习框架。**

Fail2Fix-RL 是一个面向小模型数学推理的轻量级研究框架，用来训练模型从自己的失败 rollout 中学习纠错。当前实现围绕 GSM8K 风格可验证数学题、Qwen2.5-0.5B-Instruct 级别模型和单卡训练环境构建。

项目位于 self-improving language models、自蒸馏、纠偏学习和可验证奖励学习的交叉处。仓库代码按“可复现框架”组织，不包含临时实验脚本、阶段性日志或结果堆料。

## 框架图

![Teacher-guided F2F framework](assets/f2f_framework.png)

整体分成两个阶段：

1. **离线教师纠偏**：先让学生模型采样 rollout，筛选失败或低通过率解答，再调用强教师模型诊断并修正，最后只保留通过 verifier 的纠错样本。
2. **在线 F2F 训练**：同一个 RL step 中同时生成普通解题 rollout 和纠错 rollout，用可验证奖励、纠错 replay、动态 anchor 比例和风险规避项更新策略模型。

## 为什么做这个

传统 RLVR 在数学题上通常只有二值奖励：最终答案对就是 1，错就是 0。这个信号客观，但会浪费失败轨迹里的细节。一个只差最后一步计算的 near-miss、一条思路基本正确但格式不合规的答案、以及完全跑偏的推理，在普通二值奖励下都可能是 0。

Fail2Fix-RL 把模型自己生成的失败答案重新变成训练材料。学生模型先正常解题，然后把部分候选答案作为“可能有错”的参考轨迹重新输入，让模型学习检查、保留或修正。

## 为什么需要教师引导

加入 teacher-guided 阶段的直接原因是：对极小模型来说，完全无指导的自我纠偏冷启动非常困难。我们在预实验中让模型直接从自己的 rollout 构造 correction prompt，并且只依赖二值 verifier 奖励学习。结果 correction reward 前期确实会升高，但训练中段出现明显下跌，后续长期处在高噪声震荡状态，没有形成稳定收敛。raw correction reward 也呈现同样现象：早期改善，随后严重回落，并伴随明显波动。

| 无教师引导的 correction reward | 无教师引导的 raw correction reward |
| --- | --- |
| ![Unguided correction reward mean](assets/unguided_correction_reward_mean.png) | ![Unguided correction raw reward mean](assets/unguided_correction_raw_reward_mean.png) |

在经过教师纠错初始化之后，预实验曲线呈现出明显不同的训练动态。base reward 在前 500 step 内整体呈逐步上升趋势；correction raw reward 虽然仍有在线 RL 常见的噪声，但大部分时间维持在更高区间，没有出现无教师冷启动中那种明显塌陷。

| 教师引导后的 base reward | 教师引导后的 correction raw reward |
| --- | --- |
| ![Teacher-guided base reward mean](assets/teacher_guided_base_reward_mean.png) | ![Teacher-guided correction raw reward mean](assets/teacher_guided_correction_raw_reward_mean.png) |

这种对比就是 teacher-guided 设计的主要动机。对 sub-billion 级别学生模型来说，它自己的失败 rollout 往往太噪，不能稳定作为训练材料；二值 verifier 只能告诉模型最终答案对不对，却不能告诉它推理应该怎么修；而 correction prompt 还可能诱发 over-editing，也就是把原本正确的解答改错。Teacher-guided correction SFT 的作用不是替代在线 RL，而是在 RL 之前给小模型一个最低限度的纠偏先验：如何定位错误、什么时候应该保留正确解法、怎样输出 verifier 友好的最终答案，以及如何把失败轨迹修成正确轨迹。之后的在线 F2F 仍然继续使用学生模型自己的 rollout 来训练，而不是每一步都依赖教师模型。

## 预实验结果

![Teacher-guided F2F eval summary](assets/f2f_eval_summary.png)

在 GSM8K eval 上，当前预实验中观测到的最好 accuracy 分别是：Base model 36.5%、GRPO 47.2%、无教师 F2F 48.7%、Teacher-Guided F2F 51.5%。但只看最好值还不够：无教师 F2F 虽然短暂提升，但后期会从 48.7% 跌到 39.5%；Teacher-Guided F2F 相比塌陷后的 F2F 恢复了 +12.0 个百分点，并在这个 pilot 对比中取得了最高 eval 分数，同时缓解了小模型无指导自纠偏的不稳定问题。

## 公开的教师纠偏数据

本仓库已包含用于 teacher-guided 初始化的 Mimo 纠偏数据集：

```text
released_data/mimo_teacher_corrections_gsm8k/
```

该数据集包含 3,676 条 verified correction SFT 样本，由 `mimo-v2.5-pro` 基于 GSM8K 学生失败解答生成。每条样本包含原始题目、学生失败解答、教师结构化错误诊断，以及通过 verifier 的纠偏目标。普通 SFT 只需要使用其中的 `prompt` 和 `response` 字段。

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
  f2f_eval_summary.png              预实验 eval 汇总图。

released_data/
  mimo_teacher_corrections_gsm8k/   Verified Mimo correction SFT 数据。

src/
  prepare_gsm8k_grpo_data.py        将 ModelScope GSM8K 转成 RL JSONL。
  collect_student_rollouts.py       采样多条 student rollout。
  build_teacher_corrections.py      构造 verified teacher correction SFT 数据。
  train_correction_sft.py           全参数 correction SFT。
  train_f2f_online_rl.py            在线 F2F RL 主训练脚本。
  train_grpo_base.py                Vanilla GRPO baseline。
  eval_gsm8k_subset.py              GSM8K 子集评测。
  eval_correction_sft.py            纠错 prompt 评测。
  verifier_math.py                  数学答案抽取与等价比较。

requirements.txt                    Python 依赖。
```

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

准备 GSM8K：

```bash
python src/prepare_gsm8k_grpo_data.py \
  --output-dir data/gsm8k_grpo
```

采样学生模型 rollout：

```bash
python src/collect_student_rollouts.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data data/gsm8k_grpo/train.jsonl \
  --output data/teacher_correction/student_rollouts_train.jsonl \
  --limit 512 \
  --group-size 8
```

构造教师纠错数据：

```bash
cp .env.example .env.teacher

python src/build_teacher_corrections.py \
  --rollouts data/teacher_correction/student_rollouts_train.jsonl \
  --output data/teacher_correction/correction_sft_train.jsonl \
  --cache data/teacher_correction/teacher_cache.jsonl
```

运行 correction SFT：

```bash
python src/train_correction_sft.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data data/teacher_correction/correction_sft_train.jsonl \
  --output-dir checkpoints/f2f_correction_sft
```

继续在线 F2F RL：

```bash
python src/train_f2f_online_rl.py \
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
python src/train_grpo_base.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data data/gsm8k_grpo/train.jsonl \
  --output-dir checkpoints/grpo_base \
  --report-to-tensorboard
```

评测 checkpoint：

```bash
python src/eval_gsm8k_subset.py \
  --model checkpoints/f2f_online/best_eval_checkpoint \
  --data data/gsm8k_grpo/test.jsonl \
  --limit 200 \
  --output-dir reports/eval_gsm8k
```

## 相关工作

Fail2Fix-RL 更接近自生成监督、自我纠偏和可验证奖励训练的交叉方向。

**自生成数据与自蒸馏。** Self-Instruct 说明了语言模型可以生成 instruction 数据，再用于后续 instruction tuning（[Wang et al., 2022](https://arxiv.org/abs/2212.10560)）。Large Language Models Can Self-Improve 使用 unlabeled data 上高置信度的 chain-of-thought 样本作为自训练目标（[Huang et al., 2022](https://arxiv.org/abs/2210.11610)）。STaR 通过“生成 rationale、保留答案正确的 rationale、再 fine-tune”的循环来 bootstrap reasoning（[Zelikman et al., 2022](https://arxiv.org/abs/2203.14465)）。ReST-EM 把 generate-filter-finetune 扩展到带 scalar feedback 的数学和代码问题求解（[Singh et al., 2023](https://arxiv.org/abs/2312.06585)）。SPIN 将自我提升建模成与旧版本模型生成结果之间的 self-play fine-tuning（[Chen et al., 2024](https://arxiv.org/abs/2401.01335)）。Instruction Backtranslation 则从原始文本反向构造 instruction-response 数据，是另一类 self-alignment 路线（[Li et al., 2023](https://arxiv.org/abs/2308.06259)）。

**自我纠偏与反馈驱动修正。** Self-Refine 在推理时通过 self-feedback 迭代改写输出，不直接更新模型权重（[Madaan et al., 2023](https://arxiv.org/abs/2303.17651)）。Reflexion 把任务反馈转成 verbal memory，用于下一轮 agent 尝试，尤其适合代码和决策任务（[Shinn et al., 2023](https://arxiv.org/abs/2303.11366)）。Let's Verify Step by Step 这类 process-supervision 工作说明，只看最终答案的 outcome supervision 对多步推理来说往往太粗，verifier 或 reward-model feedback 可以提供更细的学习信号（[Lightman et al., 2023](https://arxiv.org/abs/2305.20050)）。

**Failure-to-fix RL。** Fail2Fix-RL 与纯 self-training 的区别在于，它会把学生模型自己的候选解答显式 replay 成纠错 prompt。教师纠错先给小模型一个 self-correction prior，在线 F2F RL 再持续从当前策略生成新的普通 rollout 和纠错 rollout。也有更直接面向 failure trajectory correction 的 RLVR 工作研究如何把失败轨迹转成纠错监督（[Ren et al., 2026](https://arxiv.org/abs/2605.14539)）；本仓库把这一路线作为背景，主要强调 teacher-guided F2F 的训练流程。

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
