#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/learning_from_failure_exp

if [[ -f .env.teacher ]]; then
  set -a
  source .env.teacher
  set +a
fi

: "${TEACHER_API_KEY:?Set TEACHER_API_KEY or create /root/autodl-tmp/learning_from_failure_exp/.env.teacher}"

OUT="checkpoints/qwen25_05b_gsm8k_teacher_cipo_smoke_v2"
mkdir -p logs

/root/miniconda3/bin/python src/train_cipo_online_teacher_grpo.py \
  --model /root/models \
  --train-data data/gsm8k_grpo/train.jsonl \
  --eval-data data/gsm8k_grpo/test.jsonl \
  --output-dir "$OUT" \
  --train-limit 64 \
  --eval-limit 8 \
  --max-steps 3 \
  --batch-size 4 \
  --group-size 4 \
  --replay-fraction 1.0 \
  --generation-batch-size 2 \
  --forward-batch-size 2 \
  --max-prompt-length 1024 \
  --max-new-tokens 1024 \
  --eval-max-prompt-length 1024 \
  --eval-max-new-tokens 1024 \
  --temperature 0.7 \
  --top-p 0.95 \
  --lr 5e-7 \
  --beta 1e-4 \
  --clip-range 0.2 \
  --correction-lambda 1.0 \
  --risk-lambda 1.0 \
  --rho0 0.4 \
  --rho-min 0.35 \
  --rho-max 0.8 \
  --retention-target 0.85 \
  --teacher-online \
  --teacher-api-base "${TEACHER_API_BASE:-https://api.openai.com/v1}" \
  --teacher-api-key-env TEACHER_API_KEY \
  --teacher-model "${TEACHER_MODEL:-teacher-model}" \
  --teacher-trigger-pass-rate 0.25 \
  --teacher-max-calls-per-step 1 \
  --teacher-lambda 0.05 \
  --teacher-max-tokens 4096 \
  --teacher-sft-max-tokens 2048 \
  --eval-steps 3 \
  --logging-steps 1 \
  --seed 20260530 \
  --tensorboard
