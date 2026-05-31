#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/learning_from_failure_exp

RUN_NAME="${RUN_NAME:-teacher_correction_pilot_n64_g8_m1024_t32_sft20}"
MODEL="${MODEL:-/root/models}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
TRAIN_DATA="${TRAIN_DATA:-/root/autodl-tmp/learning_from_failure_exp/data/gsm8k_grpo/train.jsonl}"
BASE_DIR="${BASE_DIR:-/root/autodl-tmp/learning_from_failure_exp/data/teacher_correction/${RUN_NAME}}"
ROLLOUTS="${ROLLOUTS:-${BASE_DIR}/student_rollouts.jsonl}"
CORRECTIONS="${CORRECTIONS:-${BASE_DIR}/correction_sft.jsonl}"
CACHE="${CACHE:-/root/autodl-tmp/learning_from_failure_exp/data/teacher_correction/teacher_cache.jsonl}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/learning_from_failure_exp/logs}"
CKPT_DIR="${CKPT_DIR:-/root/autodl-tmp/learning_from_failure_exp/checkpoints/${RUN_NAME}}"
TB_DIR="${TB_DIR:-/root/autodl-tmp/learning_from_failure_exp/tensorboard/${RUN_NAME}}"

mkdir -p "${BASE_DIR}" "${LOG_DIR}" "${CKPT_DIR}" "${TB_DIR}"

echo "[1/3] collecting student rollouts -> ${ROLLOUTS}"
"${PYTHON}" src/collect_student_rollouts.py \
  --model "${MODEL}" \
  --data "${TRAIN_DATA}" \
  --output "${ROLLOUTS}" \
  --limit "${ROLLOUT_LIMIT:-64}" \
  --group-size "${GROUP_SIZE:-8}" \
  --prompt-batch-size "${PROMPT_BATCH_SIZE:-1}" \
  --generation-batch-size "${GENERATION_BATCH_SIZE:-4}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-1024}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --top-p "${TOP_P:-0.95}" \
  --seed "${SEED:-30}" \
  2>&1 | tee "${LOG_DIR}/${RUN_NAME}_collect.log"

echo "[2/3] building teacher corrections -> ${CORRECTIONS}"
"${PYTHON}" src/build_teacher_corrections.py \
  --rollouts "${ROLLOUTS}" \
  --output "${CORRECTIONS}" \
  --cache "${CACHE}" \
  --env-file "${ENV_FILE:-/root/autodl-tmp/learning_from_failure_exp/.env.teacher}" \
  --max-samples "${TEACHER_MAX_SAMPLES:-32}" \
  --max-pass-rate "${MAX_PASS_RATE:-0.25}" \
  --teacher-api-base "${TEACHER_API_BASE:-https://api.openai.com/v1}" \
  --teacher-max-tokens "${TEACHER_MAX_TOKENS:-4096}" \
  --teacher-model "${TEACHER_MODEL:-teacher-model}" \
  --teacher-concurrency "${TEACHER_CONCURRENCY:-128}" \
  2>&1 | tee "${LOG_DIR}/${RUN_NAME}_teacher.log"

echo "[3/3] correction SFT smoke -> ${CKPT_DIR}"
"${PYTHON}" src/train_correction_sft.py \
  --model "${MODEL}" \
  --data "${CORRECTIONS}" \
  --output-dir "${CKPT_DIR}" \
  --logging-dir "${TB_DIR}" \
  --max-length "${SFT_MAX_LENGTH:-4096}" \
  --batch-size "${SFT_BATCH_SIZE:-2}" \
  --grad-accum "${SFT_GRAD_ACCUM:-8}" \
  --lr "${SFT_LR:-1e-5}" \
  --max-steps "${SFT_MAX_STEPS:-20}" \
  --save-steps "${SFT_SAVE_STEPS:-20}" \
  --save-total-limit 1 \
  --logging-steps 1 \
  2>&1 | tee "${LOG_DIR}/${RUN_NAME}_sft.log"

echo "done"
