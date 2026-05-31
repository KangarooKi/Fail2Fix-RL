#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/learning_from_failure_exp

if [[ -f .env.teacher ]]; then
  set -a
  source .env.teacher
  set +a
fi

: "${TEACHER_API_KEY:?Set TEACHER_API_KEY or create /root/autodl-tmp/learning_from_failure_exp/.env.teacher}"

OUT="checkpoints/qwen25_05b_gsm8k_teacher_cipo_g8_b16_m1024_s300_rho035_t2_v2"
LOG="logs/qwen25_05b_gsm8k_teacher_cipo_g8_b16_m1024_s300_rho035_t2_v2.log"
mkdir -p logs

/root/miniconda3/bin/python src/train_cipo_online_teacher_grpo.py \
  --model /root/models \
  --train-data data/gsm8k_grpo/train.jsonl \
  --eval-data data/gsm8k_grpo/test.jsonl \
  --output-dir "$OUT" \
  --train-limit 0 \
  --eval-limit 64 \
  --eval-offset 0 \
  --max-steps 300 \
  --batch-size 16 \
  --group-size 8 \
  --replay-fraction 1.0 \
  --generation-batch-size 4 \
  --forward-batch-size 4 \
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
  --rho-w1 0.8 \
  --rho-w2 0.3 \
  --rho-w3 0.05 \
  --delta-low 0.375 \
  --delta-high 0.75 \
  --teacher-online \
  --teacher-api-base "${TEACHER_API_BASE:-https://api.openai.com/v1}" \
  --teacher-api-key-env TEACHER_API_KEY \
  --teacher-model "${TEACHER_MODEL:-teacher-model}" \
  --teacher-trigger-pass-rate 0.25 \
  --teacher-max-calls-per-step 2 \
  --teacher-lambda 0.05 \
  --teacher-max-tokens 4096 \
  --teacher-sft-max-tokens 2048 \
  --eval-steps 50 \
  --logging-steps 1 \
  --seed 20260530 \
  --tensorboard 2>&1 | tee "$LOG"
