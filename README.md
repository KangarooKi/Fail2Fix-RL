# Fail2Fix-RL

**Teacher-guided Failure-to-Fix reinforcement learning for small reasoning models.**

Fail2Fix-RL is a compact research framework for training small reasoning models to repair their own failed rollouts. The current implementation targets GSM8K-style verifiable math reasoning, Qwen2.5-0.5B-Instruct-scale models, and single-GPU experiments.

The project is inspired by correction-oriented RLVR methods such as CIPO, but the public code is organized as a clean, runnable framework rather than a dump of one-off experiment scripts.

[中文介绍](README.zh-CN.md)

## Framework

![Teacher-guided F2F framework](assets/f2f_framework.png)

The framework has two stages:

1. **Offline teacher correction**: sample student rollouts, select failed or low-pass-rate attempts, ask a strong teacher model to diagnose and fix them, and keep only corrections that pass the deterministic verifier.
2. **Online F2F training**: run base problem-solving rollouts and correction rollouts in the same RL loop, then optimize the policy with verifiable rewards, correction replay, adaptive anchor selection, and risk-aware penalties.

## Why This Exists

RLVR on math tasks often reduces feedback to a sparse binary reward: the final answer is either correct or incorrect. That signal is objective, but it ignores useful structure inside failed reasoning traces. A near miss, a small arithmetic slip, and a fully irrelevant solution all receive the same score.

Fail2Fix-RL turns model-generated failures into training material. The student first solves the problem normally. Then selected candidate solutions are replayed as "possibly wrong" traces, and the student learns to check, preserve, or repair them.

## Core Method

Each online RL step uses two streams:

```text
problem
  -> base rollouts
  -> verifier
  -> grouped RL advantage

problem + candidate solution
  -> correction rollouts
  -> verifier
  -> F2F correction reward + risk-aware shaping
```

Key mechanisms:

- **Failure-to-fix replay**: correction prompts are built from the current policy's own rollouts.
- **Teacher-guided initialization**: verified teacher corrections can be used for SFT before online RL.
- **Difficulty-aware selection**: mixed-success prompts are prioritized because they carry stronger learning signal.
- **Adaptive rho control**: the correct-vs-failed anchor ratio changes with correction retention.
- **Risk-aware reward shaping**: corrupting an already-correct solution is penalized.
- **Deterministic verifier**: final answers are extracted and compared with numeric/symbolic equivalence rules.

## Repository Scope

This repository keeps only reusable framework code: data preparation, rollout collection, teacher correction generation, correction SFT, online F2F RL, GRPO baseline training, and evaluation. Checkpoints, raw datasets, TensorBoard logs, run-specific monitors, ad hoc analysis scripts, and result tables are intentionally excluded so final paper/PPT artifacts can be curated separately.

## Repository Layout

```text
assets/
  f2f_framework.png                 Framework diagram.

remote_scripts/
  prepare_gsm8k_grpo_data.py        Convert ModelScope GSM8K into RL JSONL.
  collect_student_rollouts.py       Generate multi-sample student rollouts.
  build_teacher_corrections.py      Build verified teacher correction SFT data.
  train_correction_sft.py           Full-parameter correction SFT.
  train_f2f_online_rl.py            Online F2F RL training loop.
  train_grpo_base.py                Vanilla GRPO baseline.
  eval_gsm8k_subset.py              Deterministic GSM8K subset evaluation.
  eval_correction_sft.py            Correction-prompt evaluation.

verifier_math.py                    Answer extraction and equivalence checks.
requirements.txt                    Python dependencies.
```

## Quick Start

Install dependencies in the training environment:

```bash
pip install -r requirements.txt
```

Prepare GSM8K:

```bash
python remote_scripts/prepare_gsm8k_grpo_data.py \
  --output-dir data/gsm8k_grpo
```

Collect student rollouts:

```bash
python remote_scripts/collect_student_rollouts.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data data/gsm8k_grpo/train.jsonl \
  --output data/teacher_correction/student_rollouts_train.jsonl \
  --limit 512 \
  --group-size 8
```

Generate verified teacher corrections:

```bash
cp .env.example .env.teacher

python remote_scripts/build_teacher_corrections.py \
  --rollouts data/teacher_correction/student_rollouts_train.jsonl \
  --output data/teacher_correction/correction_sft_train.jsonl \
  --cache data/teacher_correction/teacher_cache.jsonl
```

Run correction SFT:

```bash
python remote_scripts/train_correction_sft.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data data/teacher_correction/correction_sft_train.jsonl \
  --output-dir checkpoints/f2f_correction_sft
```

Continue with online F2F RL:

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

Run the GRPO baseline:

```bash
python remote_scripts/train_grpo_base.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data data/gsm8k_grpo/train.jsonl \
  --output-dir checkpoints/grpo_base \
  --report-to-tensorboard
```

Evaluate a checkpoint:

```bash
python remote_scripts/eval_gsm8k_subset.py \
  --model checkpoints/f2f_online/best_eval_checkpoint \
  --data data/gsm8k_grpo/test.jsonl \
  --limit 200 \
  --output-dir reports/eval_gsm8k
```

## Reproducibility Notes

- Training scripts assume a CUDA environment for practical speed.
- Full-parameter Qwen2.5-0.5B runs were developed on a 48GB RTX 4090 setup.
- API keys should only live in local environment files such as `.env.teacher`.
- Raw datasets, checkpoints, generated rollouts, logs, and TensorBoard event files are ignored by git.

## Citation

If you use this project in a report or presentation, cite it as an experimental framework inspired by correction-oriented RLVR methods:

```text
Fail2Fix-RL: Teacher-guided Failure-to-Fix Reinforcement Learning.
Research prototype, 2026.
```

Related paper:

```text
Correction Intention Preference Optimization
arXiv:2605.14539
https://arxiv.org/abs/2605.14539
```
