# Fail2Fix-RL

**Learning to correct from failed reasoning rollouts.**

Fail2Fix-RL is a compact research framework for training small reasoning models with verifiable rewards and correction replay. It was built around GSM8K-style math reasoning, Qwen2.5-0.5B-Instruct, and single-GPU experiments where full-scale RL infrastructure is too heavy.

The project is inspired by correction-oriented RLVR methods such as CIPO, but it is not presented as a new paper implementation claim. The goal is practical: make failure-driven self-correction experiments runnable, inspectable, and easy to iterate on.

[中文介绍](README.zh-CN.md)

## Why This Exists

RLVR on math tasks often gives sparse binary feedback: a rollout is either correct or incorrect. That signal is objective, but it wastes structure inside failed attempts. A near miss, an arithmetic slip, and an irrelevant solution all receive the same reward.

Fail2Fix-RL treats model-generated solutions as training material. The model first solves a problem normally. Then selected candidate solutions are replayed back as "possibly wrong" traces, and the model is trained to check, preserve, or repair them.

## Method

Each online RL step has two streams:

```text
original problem
  -> base rollouts
  -> deterministic math verifier
  -> GRPO-style grouped advantages

selected candidate solution
  -> correction prompt
  -> correction rollouts
  -> same verifier
  -> correction objective + risk-aware shaping
```

Core mechanisms:

- **Online correction replay**: correction prompts are built from the current policy's own rollouts.
- **Difficulty-aware selection**: prompts with mixed success/failure rollouts are prioritized.
- **Adaptive rho control**: the success-vs-failure anchor ratio is adjusted using correction retention.
- **Risk-aware reward shaping**: revising a correct solution into a wrong one receives an extra penalty.
- **Teacher correction SFT**: an optional teacher-generated correction dataset can initialize the student before RL.
- **Deterministic verifier**: final answers are extracted and compared with numeric/symbolic equivalence rules.
- **Single-GPU monitoring**: TensorBoard logging and OOM/recovery helpers are included for long runs.

## Current Pipeline

The main experimental path is:

1. Prepare GSM8K train/test JSONL files.
2. Collect student rollouts from Qwen2.5-0.5B-Instruct.
3. Use a strong teacher model to build verified correction examples.
4. Run full-parameter correction SFT on the small model.
5. Continue with online CIPO/Fail2Fix-RL using base and correction streams.
6. Evaluate checkpoints on fixed GSM8K subsets.

## What Is Included

This initial repository release contains the training framework code, data preparation utilities, evaluation scripts, and monitoring helpers. Experiment result tables, plots, checkpoints, raw datasets, and long run logs are intentionally kept out of the repository so they can be curated separately.

The code supports four main experiment families:

- Base model evaluation on GSM8K-style verifiable math data.
- Vanilla GRPO training as the RLVR baseline.
- Plain F2F/CIPO-style correction replay training.
- Teacher-guided F2F, where verified teacher corrections initialize the student before online self-correction RL.

## Repository Layout

```text
remote_scripts/
  prepare_gsm8k_grpo_data.py        Convert ModelScope GSM8K into RL JSONL.
  collect_student_rollouts.py       Generate multi-sample student rollouts.
  build_teacher_corrections.py      Build verified teacher correction SFT data.
  train_correction_sft.py           Full-parameter correction SFT.
  train_cipo_online_grpo.py         Online Fail2Fix/CIPO-style RL loop.
  train_dapo_grpo_base.py           Vanilla GRPO baseline.
  eval_gsm8k_subset.py              Deterministic GSM8K subset evaluation.
  eval_correction_sft.py            Correction-prompt evaluation.
  monitor_final3676_cipo.py         OOM/recovery monitor for long CIPO runs.
  analyze_gsm8k_rollout_errors.py   Rollout-level error diagnosis.

reports/
  fail2fix_summary/                 Figures and tables for early experiments.

experiment_notes/
  teacher_correction_stage2.md      Detailed stage-2 experiment log.

verifier_math.py                    Math answer extraction and equivalence checks.
```

## Quick Start

Install dependencies in your training environment:

```bash
pip install -r requirements.txt
```

Prepare GSM8K data:

```bash
python remote_scripts/prepare_gsm8k_grpo_data.py \
  --output-dir data/gsm8k_grpo
```

Run a small online CIPO/Fail2Fix smoke experiment:

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

For teacher-correction data generation, create an environment file from `.env.example` and keep the real key out of git:

```bash
cp .env.example .env.teacher
```

## Notes on Reproducibility

- The scripts assume a CUDA machine for model training.
- Full-parameter runs were developed on a 48GB RTX 4090 environment.
- Checkpoints, raw datasets, logs, and TensorBoard event files are intentionally excluded from git.
- Reported metrics use a strict verifier unless otherwise stated.
- The active experiments are still evolving; prefer `experiment_notes/` for exact run histories.

## Citation

If you use this project in a report or presentation, cite it as an experimental implementation inspired by correction-oriented RLVR/CIPO-style methods.

```text
Fail2Fix-RL: Learning to Correct from Failed Reasoning Rollouts.
Research prototype, 2026.
```

Related paper:

```text
Correction Intention Preference Optimization
arXiv:2605.14539
https://arxiv.org/abs/2605.14539
```
