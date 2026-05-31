#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/learning_from_failure_exp

PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
RUN_NAME="${RUN_NAME:-teacher_correction_stage1_train512_eval128_g8_m1024_t256_sft200}"
MODEL="${MODEL:-/root/models}"
TRAIN_DATA="${TRAIN_DATA:-/root/autodl-tmp/learning_from_failure_exp/data/gsm8k_grpo/train.jsonl}"
EVAL_DATA="${EVAL_DATA:-/root/autodl-tmp/learning_from_failure_exp/data/gsm8k_grpo/test.jsonl}"

BASE_DIR="${BASE_DIR:-/root/autodl-tmp/learning_from_failure_exp/data/teacher_correction/${RUN_NAME}}"
TRAIN_ROLLOUTS="${TRAIN_ROLLOUTS:-${BASE_DIR}/train_student_rollouts.jsonl}"
EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-${BASE_DIR}/eval_student_rollouts.jsonl}"
CORRECTIONS="${CORRECTIONS:-${BASE_DIR}/correction_sft_train.jsonl}"
CACHE="${CACHE:-/root/autodl-tmp/learning_from_failure_exp/data/teacher_correction/teacher_cache.jsonl}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/learning_from_failure_exp/logs}"
CKPT_DIR="${CKPT_DIR:-/root/autodl-tmp/learning_from_failure_exp/checkpoints/${RUN_NAME}}"
TB_DIR="${TB_DIR:-/root/autodl-tmp/learning_from_failure_exp/tensorboard/${RUN_NAME}}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/learning_from_failure_exp/reports/${RUN_NAME}}"

mkdir -p "${BASE_DIR}" "${LOG_DIR}" "${CKPT_DIR}" "${TB_DIR}" "${REPORT_DIR}"

echo "[1/5] collect train student rollouts -> ${TRAIN_ROLLOUTS}"
"${PYTHON}" src/collect_student_rollouts.py \
  --model "${MODEL}" \
  --data "${TRAIN_DATA}" \
  --output "${TRAIN_ROLLOUTS}" \
  --limit "${TRAIN_ROLLOUT_LIMIT:-512}" \
  --offset "${TRAIN_ROLLOUT_OFFSET:-64}" \
  --group-size "${GROUP_SIZE:-8}" \
  --prompt-batch-size "${PROMPT_BATCH_SIZE:-128}" \
  --generation-batch-size "${GENERATION_BATCH_SIZE:-4}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-1024}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --top-p "${TOP_P:-0.95}" \
  --seed "${SEED:-31}" \
  2>&1 | tee -a "${LOG_DIR}/${RUN_NAME}_01_collect_train.log"

echo "[2/5] collect held-out eval student rollouts -> ${EVAL_ROLLOUTS}"
"${PYTHON}" src/collect_student_rollouts.py \
  --model "${MODEL}" \
  --data "${EVAL_DATA}" \
  --output "${EVAL_ROLLOUTS}" \
  --limit "${EVAL_ROLLOUT_LIMIT:-128}" \
  --offset "${EVAL_ROLLOUT_OFFSET:-0}" \
  --group-size "${GROUP_SIZE:-8}" \
  --prompt-batch-size "${PROMPT_BATCH_SIZE:-128}" \
  --generation-batch-size "${GENERATION_BATCH_SIZE:-4}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-1024}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --top-p "${TOP_P:-0.95}" \
  --seed "${SEED:-31}" \
  2>&1 | tee -a "${LOG_DIR}/${RUN_NAME}_02_collect_eval.log"

echo "[3/5] build teacher correction SFT data -> ${CORRECTIONS}"
"${PYTHON}" src/build_teacher_corrections.py \
  --rollouts "${TRAIN_ROLLOUTS}" \
  --output "${CORRECTIONS}" \
  --cache "${CACHE}" \
  --env-file "${ENV_FILE:-/root/autodl-tmp/learning_from_failure_exp/.env.teacher}" \
  --max-samples "${TEACHER_MAX_SAMPLES:-256}" \
  --max-pass-rate "${MAX_PASS_RATE:-0.25}" \
  --teacher-api-base "${TEACHER_API_BASE:-https://api.openai.com/v1}" \
  --teacher-max-tokens "${TEACHER_MAX_TOKENS:-4096}" \
  --teacher-model "${TEACHER_MODEL:-teacher-model}" \
  --teacher-concurrency "${TEACHER_CONCURRENCY:-128}" \
  2>&1 | tee -a "${LOG_DIR}/${RUN_NAME}_03_teacher.log"

echo "[4/5] correction SFT -> ${CKPT_DIR}"
"${PYTHON}" src/train_correction_sft.py \
  --model "${MODEL}" \
  --data "${CORRECTIONS}" \
  --output-dir "${CKPT_DIR}" \
  --logging-dir "${TB_DIR}" \
  --max-length "${SFT_MAX_LENGTH:-4096}" \
  --batch-size "${SFT_BATCH_SIZE:-2}" \
  --grad-accum "${SFT_GRAD_ACCUM:-8}" \
  --lr "${SFT_LR:-1e-5}" \
  --max-steps "${SFT_MAX_STEPS:-200}" \
  --save-steps "${SFT_SAVE_STEPS:-100000}" \
  --save-total-limit 1 \
  --logging-steps "${SFT_LOGGING_STEPS:-5}" \
  2>&1 | tee -a "${LOG_DIR}/${RUN_NAME}_04_sft.log"

echo "[5/5] held-out correction eval: base"
"${PYTHON}" src/eval_correction_sft.py \
  --model "${MODEL}" \
  --rollouts "${EVAL_ROLLOUTS}" \
  --output-dir "${REPORT_DIR}" \
  --limit "${CORRECTION_EVAL_LIMIT:-128}" \
  --max-pass-rate "${MAX_PASS_RATE:-0.25}" \
  --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-768}" \
  2>&1 | tee -a "${LOG_DIR}/${RUN_NAME}_05_eval_base.log"

echo "[5/5] held-out correction eval: sft"
"${PYTHON}" src/eval_correction_sft.py \
  --model "${CKPT_DIR}" \
  --rollouts "${EVAL_ROLLOUTS}" \
  --output-dir "${REPORT_DIR}" \
  --limit "${CORRECTION_EVAL_LIMIT:-128}" \
  --max-pass-rate "${MAX_PASS_RATE:-0.25}" \
  --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-768}" \
  2>&1 | tee -a "${LOG_DIR}/${RUN_NAME}_06_eval_sft.log"

echo "done"
