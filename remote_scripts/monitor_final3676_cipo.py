import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path


PROJECT_DIR = Path("/root/autodl-tmp/learning_from_failure_exp")
PYTHON = "/root/miniconda3/bin/python"
TENSORBOARD = "/root/miniconda3/bin/tensorboard"

RUN_BASE = "qwen25_05b_teacher_correction_final3676_cipo_online_g8_b16"
RUN_SUFFIX = "s500_20260530"
INITIAL_MODEL = PROJECT_DIR / "checkpoints/teacher_correction_stage2_final3676_sft1000_20260530_214604"
INITIAL_OUTPUT_DIR = PROJECT_DIR / "checkpoints/qwen25_05b_teacher_correction_final3676_cipo_online_g8_b16_m1024_s500_20260530"
INITIAL_LOG_PATH = PROJECT_DIR / "logs/qwen25_05b_teacher_correction_final3676_cipo_online_g8_b16_m1024_s500_20260530.log"

STATE_PATH = PROJECT_DIR / "logs/final3676_cipo_oom_monitor_state.json"
MONITOR_LOG = PROJECT_DIR / "logs/final3676_cipo_oom_monitor.log"

TARGET_STEPS = 500
TOKEN_LADDER = [1024, 768, 512]
TENSORBOARD_PORT = 6028
HIGH_MEM_RATIO = float(os.environ.get("FINAL3676_CIPO_HIGH_MEM_RATIO", "0.94"))
HIGH_MEM_PATIENCE = int(os.environ.get("FINAL3676_CIPO_HIGH_MEM_PATIENCE", "3"))
POLL_SECONDS = int(os.environ.get("FINAL3676_CIPO_MONITOR_POLL_SECONDS", "30"))

OOM_PAT = re.compile(
    r"(cuda out of memory|outofmemoryerror|out of memory|cublas_status_alloc_failed|cuda error: out of memory)",
    re.IGNORECASE,
)


def log(message, **fields):
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "message": message}
    rec.update(fields)
    MONITOR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with MONITOR_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def run(cmd):
    return subprocess.run(
        cmd,
        cwd=str(PROJECT_DIR),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def default_state():
    return {
        "status": "monitoring",
        "current_max_tokens": 1024,
        "restart_count": 0,
        "output_dir": str(INITIAL_OUTPUT_DIR),
        "log_path": str(INITIAL_LOG_PATH),
        "resume_from_step": 0,
        "target_steps": TARGET_STEPS,
        "high_mem_count": 0,
        "oom_events": [],
        "restart_events": [],
    }


def load_state():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return default_state()


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def output_dir_for(max_tokens, restart_count):
    suffix = f"{RUN_BASE}_m{max_tokens}_{RUN_SUFFIX}"
    if restart_count:
        suffix += f"_recover{restart_count}"
    return PROJECT_DIR / "checkpoints" / suffix


def log_path_for(max_tokens, restart_count):
    suffix = f"{RUN_BASE}_m{max_tokens}_{RUN_SUFFIX}"
    if restart_count:
        suffix += f"_recover{restart_count}"
    return PROJECT_DIR / "logs" / f"{suffix}.log"


def list_processes():
    proc = run(["ps", "-eo", "pid=,args="])
    rows = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, args = line.partition(" ")
        try:
            rows.append((int(pid_text), args))
        except ValueError:
            continue
    return rows


def train_pids_for_output(output_dir):
    output_dir = Path(output_dir)
    output_abs = str(output_dir)
    try:
        output_rel = str(output_dir.relative_to(PROJECT_DIR))
    except ValueError:
        output_rel = str(output_dir)
    pids = []
    for pid, args in list_processes():
        if "train_cipo_online_grpo.py" in args and (output_abs in args or output_rel in args):
            pids.append(pid)
    return pids


def kill_pids(pids, reason):
    if not pids:
        return
    log("killing_training_process", pids=pids, reason=reason)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(8)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def read_tail(path, max_bytes=2_000_000):
    path = Path(path)
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        return f.read().decode("utf-8", errors="replace")


def has_oom(log_path):
    return bool(OOM_PAT.search(read_tail(log_path)))


def gpu_memory():
    proc = run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"])
    first = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    parts = [x.strip() for x in first.split(",")]
    if len(parts) < 3:
        return None, None, None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None, None, None


def latest_train_step(output_dir):
    path = Path(output_dir) / "train_history.jsonl"
    if not path.exists():
        return 0
    last = ""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = line
    if not last:
        return 0
    try:
        return int(json.loads(last).get("step", 0))
    except Exception:
        return 0


def latest_eval_step(output_dir):
    path = Path(output_dir) / "last_step.txt"
    if path.exists():
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except Exception:
            pass
    hist = Path(output_dir) / "eval_history.jsonl"
    if not hist.exists():
        return 0
    last = ""
    with hist.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = line
    if not last:
        return 0
    try:
        return int(json.loads(last).get("step", 0))
    except Exception:
        return 0


def choose_resume_model(output_dir):
    output_dir = Path(output_dir)
    last_ckpt = output_dir / "last_eval_checkpoint"
    if last_ckpt.exists():
        return last_ckpt, latest_eval_step(output_dir)
    return INITIAL_MODEL, 0


def next_tokens(current, reason):
    current = int(current)
    if reason == "missing_process":
        return current
    for tok in TOKEN_LADDER:
        if tok < current:
            return tok
    return current


def start_tensorboard(output_dir):
    run(["pkill", "-f", f"tensorboard .*port {TENSORBOARD_PORT}"])
    log_file = PROJECT_DIR / f"logs/tensorboard_final3676_cipo_{TENSORBOARD_PORT}.log"
    with log_file.open("ab") as f:
        subprocess.Popen(
            [
                TENSORBOARD,
                "--logdir",
                str(Path(output_dir) / "runs"),
                "--host",
                "127.0.0.1",
                "--port",
                str(TENSORBOARD_PORT),
                "--reload_interval",
                "5",
            ],
            cwd=str(PROJECT_DIR),
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def start_training(model_path, output_dir, log_path, max_tokens, max_steps):
    output_dir = Path(output_dir)
    log_path = Path(log_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON,
        "src/train_cipo_online_grpo.py",
        "--model",
        str(model_path),
        "--train-data",
        "/root/autodl-tmp/learning_from_failure_exp/data/gsm8k_grpo/train.jsonl",
        "--eval-data",
        "/root/autodl-tmp/learning_from_failure_exp/data/gsm8k_grpo/test.jsonl",
        "--output-dir",
        str(output_dir),
        "--train-limit",
        "0",
        "--eval-limit",
        "200",
        "--eval-offset",
        "0",
        "--max-steps",
        str(max_steps),
        "--batch-size",
        "16",
        "--group-size",
        "8",
        "--replay-fraction",
        "1.0",
        "--generation-batch-size",
        "4",
        "--forward-batch-size",
        "4",
        "--max-prompt-length",
        "1024",
        "--max-new-tokens",
        str(max_tokens),
        "--eval-max-prompt-length",
        "1024",
        "--eval-max-new-tokens",
        str(max_tokens),
        "--temperature",
        "0.7",
        "--top-p",
        "0.95",
        "--lr",
        "5e-7",
        "--beta",
        "1e-4",
        "--clip-range",
        "0.2",
        "--correction-lambda",
        "1.0",
        "--risk-lambda",
        "1.0",
        "--rho0",
        "0.4",
        "--rho-min",
        "0.35",
        "--rho-max",
        "0.8",
        "--retention-target",
        "0.85",
        "--rho-w1",
        "0.8",
        "--rho-w2",
        "0.3",
        "--rho-w3",
        "0.05",
        "--delta-low",
        "0.375",
        "--delta-high",
        "0.75",
        "--eval-steps",
        "50",
        "--logging-steps",
        "1",
        "--seed",
        "20260530",
        "--tensorboard",
    ]
    with log_path.open("ab") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_DIR),
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    start_tensorboard(output_dir)
    return proc.pid


def recover(state, reason, used_mb=None, total_mb=None):
    current_tokens = int(state.get("current_max_tokens", 1024))
    new_tokens = next_tokens(current_tokens, reason)
    if reason != "missing_process" and new_tokens >= current_tokens and current_tokens <= min(TOKEN_LADDER):
        state["status"] = "needs_manual_review_at_min_tokens"
        save_state(state)
        log("cannot_lower_tokens_further", reason=reason, current_tokens=current_tokens)
        return state

    pids = train_pids_for_output(state["output_dir"])
    kill_pids(pids, reason)

    resume_model, resume_step = choose_resume_model(state["output_dir"])
    remaining_steps = max(1, TARGET_STEPS - int(resume_step))
    restart_count = int(state.get("restart_count", 0)) + 1
    output_dir = output_dir_for(new_tokens, restart_count)
    log_path = log_path_for(new_tokens, restart_count)
    pid = start_training(resume_model, output_dir, log_path, new_tokens, remaining_steps)

    event = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reason": reason,
        "old_max_tokens": current_tokens,
        "new_max_tokens": new_tokens,
        "resume_model": str(resume_model),
        "resume_from_step": int(resume_step),
        "remaining_steps": int(remaining_steps),
        "new_pid": int(pid),
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "used_mb": used_mb,
        "total_mb": total_mb,
    }
    state.update(
        {
            "status": "restarted",
            "current_max_tokens": new_tokens,
            "restart_count": restart_count,
            "output_dir": str(output_dir),
            "log_path": str(log_path),
            "resume_from_step": int(resume_step),
            "high_mem_count": 0,
        }
    )
    if reason in {"log_oom_detected", "high_memory_preemptive", "process_exited_after_oom"}:
        state.setdefault("oom_events", []).append(event)
    state.setdefault("restart_events", []).append(event)
    save_state(state)
    log("training_restarted", **event)
    return state


def main():
    state = load_state()
    save_state(state)
    log("monitor_started", state=state, poll_seconds=POLL_SECONDS, high_mem_ratio=HIGH_MEM_RATIO, high_mem_patience=HIGH_MEM_PATIENCE)
    while True:
        state = load_state()
        used_mb, total_mb, util = gpu_memory()
        pids = train_pids_for_output(state["output_dir"])
        latest_step = latest_train_step(state["output_dir"])

        if latest_step >= TARGET_STEPS and not pids:
            state["status"] = "completed"
            save_state(state)
            log("training_completed", latest_step=latest_step, output_dir=state["output_dir"])
            time.sleep(POLL_SECONDS)
            continue

        if has_oom(state["log_path"]):
            state = recover(state, "log_oom_detected", used_mb, total_mb)
            time.sleep(POLL_SECONDS)
            continue

        if used_mb is not None and total_mb:
            ratio = used_mb / max(1, total_mb)
            if ratio >= HIGH_MEM_RATIO:
                state["high_mem_count"] = int(state.get("high_mem_count", 0)) + 1
                save_state(state)
                log("high_memory_warning", used_mb=used_mb, total_mb=total_mb, ratio=ratio, high_mem_count=state["high_mem_count"], pids=pids, latest_step=latest_step)
                if state["high_mem_count"] >= HIGH_MEM_PATIENCE:
                    state = recover(state, "high_memory_preemptive", used_mb, total_mb)
                    time.sleep(POLL_SECONDS)
                    continue
            elif int(state.get("high_mem_count", 0)) != 0:
                state["high_mem_count"] = 0
                save_state(state)

        if not pids and latest_step < TARGET_STEPS:
            if has_oom(state["log_path"]):
                state = recover(state, "process_exited_after_oom", used_mb, total_mb)
            else:
                state = recover(state, "missing_process", used_mb, total_mb)
            time.sleep(POLL_SECONDS)
            continue

        log("monitor_tick", pids=pids, latest_step=latest_step, used_mb=used_mb, total_mb=total_mb, util=util, output_dir=state["output_dir"], max_tokens=state["current_max_tokens"])
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
