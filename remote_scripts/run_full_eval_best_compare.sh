#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/learning_from_failure_exp

OUT_DIR="reports/eval_gsm8k_full_best_compare"
LOG_DIR="logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

CIPO_MODEL="checkpoints/qwen25_05b_gsm8k_cipo_online_paper_g8_b16_m1024_s500_gb4_fb4_manual/best_eval_checkpoint"
GRPO_MODEL="checkpoints/qwen25_05b_gsm8k_grpo_base_full_g16_m768_s7473/best_eval_checkpoint"
DATA="data/gsm8k_grpo/test.jsonl"

echo "[$(date '+%F %T')] queued full eval; waiting for CIPO training to finish"
while ps -ef | grep -q "[t]rain_cipo_online_grpo.py"; do
  step=$(tail -1 checkpoints/qwen25_05b_gsm8k_cipo_online_paper_g8_b16_m1024_s500_gb4_fb4_manual/train_history.jsonl 2>/dev/null | sed -E 's/.*"step": ([0-9]+).*/\1/' || true)
  echo "[$(date '+%F %T')] training still running; latest_step=${step:-unknown}"
  sleep 120
done

echo "[$(date '+%F %T')] training finished; starting CIPO best full eval"
echo "[$(date '+%F %T')] CIPO best_step=$(cat checkpoints/qwen25_05b_gsm8k_cipo_online_paper_g8_b16_m1024_s500_gb4_fb4_manual/best_step.txt 2>/dev/null || echo unknown)"

/root/miniconda3/bin/python src/eval_gsm8k_subset.py \
  --model "$CIPO_MODEL" \
  --data "$DATA" \
  --output-dir "$OUT_DIR/cipo_best" \
  --limit 0 \
  --offset 0 \
  --max-prompt-length 1024 \
  --max-new-tokens 1024 \
  --temperature 0.0 \
  --top-p 0.95

echo "[$(date '+%F %T')] CIPO full eval finished; starting GRPO best full eval"

/root/miniconda3/bin/python src/eval_gsm8k_subset.py \
  --model "$GRPO_MODEL" \
  --data "$DATA" \
  --output-dir "$OUT_DIR/grpo_best" \
  --limit 0 \
  --offset 0 \
  --max-prompt-length 1024 \
  --max-new-tokens 1024 \
  --temperature 0.0 \
  --top-p 0.95

echo "[$(date '+%F %T')] all full evals finished"
