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

STATE_PATH = PROJECT_DIR / "logs/cipo_online_oom_monitor_state.json"
MONITOR_LOG = PROJECT_DIR / "logs/cipo_online_oom_monitor.log"

BASE_OUTPUT_STEM = "qwen25_05b_gsm8k_cipo_online_paper_g8_b16"
RUN_SUFFIX = "s500_gb4_fb4"
TOKEN_LADDER = [8192, 4096, 2048, 1024]

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


def run(cmd, check=False):
    return subprocess.run(
        cmd,
        cwd=str(PROJECT_DIR),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def output_dir_for(max_tokens, restart_count=0):
    suffix = f"{BASE_OUTPUT_STEM}_m{max_tokens}_{RUN_SUFFIX}"
    if restart_count:
        suffix = f"{suffix}_oomr{restart_count}"
    return PROJECT_DIR / "checkpoints" / suffix


def log_path_for(max_tokens, restart_count=0):
    suffix = f"cipo_online_paper_g8_b16_m{max_tokens}_{RUN_SUFFIX}"
    if restart_count:
        suffix = f"{suffix}_oomr{restart_count}"
    return PROJECT_DIR / "logs" / f"{suffix}.log"


def default_state():
    return {
        "current_max_tokens": 8192,
        "restart_count": 0,
        "output_dir": str(output_dir_for(8192, 0)),
        "log_path": str(log_path_for(8192, 0)),
        "status": "monitoring",
        "oom_events": [],
        "high_mem_count": 0,
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
    output_path = Path(output_dir)
    output_abs = str(output_path)
    try:
        output_rel = str(output_path.relative_to(PROJECT_DIR))
    except ValueError:
        output_rel = str(output_path)
    pids = []
    for pid, args in list_processes():
        if "train_cipo_online_grpo.py" in args and (output_abs in args or output_rel in args):
            pids.append(pid)
    return pids


def kill_pids(pids):
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


def read_log_tail(path, max_bytes=2_000_000):
    path = Path(path)
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        return f.read().decode("utf-8", errors="replace")


def has_oom(path):
    return bool(OOM_PAT.search(read_log_tail(path)))


def gpu_memory_used_mb():
    proc = run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
    first = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    parts = [x.strip() for x in first.split(",")]
    if len(parts) < 2:
        return None, None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None, None


def next_max_tokens(current):
    for token in TOKEN_LADDER:
        if token < current:
            return token
    return TOKEN_LADDER[-1]


def start_tensorboard(output_dir):
    run(["pkill", "-f", "tensorboard .*port 6016"])
    log_file = PROJECT_DIR / "logs/tensorboard_cipo_online_auto_6016.log"
    with log_file.open("ab") as f:
        subprocess.Popen(
            [
                TENSORBOARD,
                "--logdir",
                str(Path(output_dir) / "runs"),
                "--host",
                "127.0.0.1",
                "--port",
                "6016",
                "--reload_interval",
                "5",
            ],
            cwd=str(PROJECT_DIR),
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def start_training(max_tokens, restart_count):
    output_dir = output_dir_for(max_tokens, restart_count)
    train_log = log_path_for(max_tokens, restart_count)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    train_log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON,
        "src/train_cipo_online_grpo.py",
        "--model",
        "/root/models",
        "--train-data",
        "data/gsm8k_grpo/train.jsonl",
        "--eval-data",
        "data/gsm8k_grpo/test.jsonl",
        "--output-dir",
        str(output_dir.relative_to(PROJECT_DIR)),
        "--train-limit",
        "0",
        "--eval-limit",
        "64",
        "--eval-offset",
        "0",
        "--max-steps",
        "500",
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
        "0.3",
        "--rho-min",
        "0.2",
        "--rho-max",
        "0.8",
        "--retention-target",
        "0.8",
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
    with train_log.open("ab") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_DIR),
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    start_tensorboard(output_dir)
    return proc.pid, output_dir, train_log


def restart_with_lower_tokens(state, reason, used_mb=None, total_mb=None):
    current = int(state["current_max_tokens"])
    lowered = next_max_tokens(current)
    if lowered >= current:
        log("already_at_min_token_limit", current_max_tokens=current, reason=reason)
        state["status"] = "oom_at_min_token_limit"
        save_state(state)
        return state

    pids = train_pids_for_output(state["output_dir"])
    if pids:
        log("killing_current_training", pids=pids, output_dir=state["output_dir"], reason=reason)
        kill_pids(pids)

    restart_count = int(state.get("restart_count", 0)) + 1
    pid, output_dir, train_log = start_training(lowered, restart_count)
    event = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reason": reason,
        "old_max_tokens": current,
        "new_max_tokens": lowered,
        "used_mb": used_mb,
        "total_mb": total_mb,
        "new_pid": pid,
        "output_dir": str(output_dir),
        "log_path": str(train_log),
    }
    state.update(
        {
            "current_max_tokens": lowered,
            "restart_count": restart_count,
            "output_dir": str(output_dir),
            "log_path": str(train_log),
            "status": "restarted_after_oom",
            "high_mem_count": 0,
        }
    )
    state.setdefault("oom_events", []).append(event)
    save_state(state)
    log("restarted_with_lower_tokens", **event)
    return state


def main():
    state = load_state()
    save_state(state)
    log("monitor_started", state=state)
    poll_seconds = int(os.environ.get("CIPO_MONITOR_POLL_SECONDS", "30"))
    high_mem_threshold_ratio = float(os.environ.get("CIPO_HIGH_MEM_RATIO", "0.94"))
    high_mem_patience = int(os.environ.get("CIPO_HIGH_MEM_PATIENCE", "0"))

    while True:
        state = load_state()
        pids = train_pids_for_output(state["output_dir"])
        used_mb, total_mb = gpu_memory_used_mb()

        if has_oom(state["log_path"]):
            state = restart_with_lower_tokens(state, "log_oom_detected", used_mb, total_mb)
            time.sleep(poll_seconds)
            continue

        if high_mem_patience > 0 and used_mb is not None and total_mb:
            ratio = used_mb / max(1, total_mb)
            if ratio >= high_mem_threshold_ratio:
                state["high_mem_count"] = int(state.get("high_mem_count", 0)) + 1
                save_state(state)
                log(
                    "high_memory_warning",
                    used_mb=used_mb,
                    total_mb=total_mb,
                    ratio=ratio,
                    high_mem_count=state["high_mem_count"],
                    pids=pids,
                )
                if state["high_mem_count"] >= high_mem_patience:
                    state = restart_with_lower_tokens(state, "high_memory_preemptive", used_mb, total_mb)
            elif int(state.get("high_mem_count", 0)) != 0:
                state["high_mem_count"] = 0
                save_state(state)

        if not pids:
            if has_oom(state["log_path"]):
                state = restart_with_lower_tokens(state, "process_exited_after_oom", used_mb, total_mb)
            else:
                log("training_process_not_found_without_oom", output_dir=state["output_dir"])

        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
