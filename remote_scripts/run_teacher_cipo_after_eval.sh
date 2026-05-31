#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/learning_from_failure_exp
mkdir -p logs

LOG="logs/teacher_cipo_after_eval_queue.log"
{
  echo "[$(date '+%F %T')] queued teacher-CIPO from-scratch run"
  echo "[$(date '+%F %T')] waiting for old CIPO training and full-eval comparison to finish"

  while true; do
    if ps -ef | grep -q "[t]rain_cipo_online_grpo.py"; then
      latest_step=$(tail -1 checkpoints/qwen25_05b_gsm8k_cipo_online_paper_g8_b16_m1024_s500_gb4_fb4_manual/train_history.jsonl 2>/dev/null | sed -E 's/.*"step": ([0-9]+).*/\1/' || true)
      echo "[$(date '+%F %T')] old CIPO still running; latest_step=${latest_step:-unknown}"
      sleep 120
      continue
    fi
    if ps -ef | grep -q "[r]un_full_eval_best_compare.sh"; then
      echo "[$(date '+%F %T')] full-eval comparison still running"
      sleep 180
      continue
    fi
    if ps -ef | grep -q "[e]val_gsm8k_subset.py"; then
      echo "[$(date '+%F %T')] eval_gsm8k_subset still running"
      sleep 180
      continue
    fi
    break
  done

  echo "[$(date '+%F %T')] dependency jobs finished; starting teacher-CIPO smoke test"
  bash src/run_teacher_cipo_smoke.sh

  echo "[$(date '+%F %T')] smoke test finished; starting teacher-CIPO full run from /root/models"
  bash src/run_teacher_cipo_full.sh

  echo "[$(date '+%F %T')] teacher-CIPO full run finished"
} >> "$LOG" 2>&1
