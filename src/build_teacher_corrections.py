import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from verifier_math import extract_final_answer, verify_math_response


SCHEMA = "teacher_correction_sft_v1"
PROMPT_VERSION = "offline_teacher_correction_v1"


def load_env_file(path):
    if not path:
        return
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_cache(path):
    cache = {}
    path = Path(path)
    if not path.exists():
        return cache
    for rec in read_jsonl(path):
        key = rec.get("key")
        if key:
            cache[key] = rec
    return cache


def append_jsonl(path, rec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_existing_keys(path):
    keys = set()
    path = Path(path)
    if not path.exists():
        return keys
    for rec in read_jsonl(path):
        key = rec.get("key")
        if key:
            keys.add(key)
    return keys


def usable_cache_record(rec, reference_answer=None):
    if not (rec and rec.get("ok") is True and str(rec.get("teacher_response", "")).strip()):
        return False
    teacher_text = rec.get("teacher_response", "")
    if reference_answer is None:
        return True
    return teacher_format_ok(teacher_text) and bool(verify_math_response(teacher_text, reference_answer)["reward"])


def correction_prompt(original_prompt, candidate_solution):
    return (
        "You are solving a grade-school math problem. A student's previous solution is provided and may be wrong.\n"
        "Check the previous solution, identify the main mistake, and write a corrected concise solution.\n\n"
        "Use exactly this output format:\n"
        "<error_type>one of: arithmetic, setup, logic, unit, counting, answer_format, other</error_type>\n"
        "<error_location>the first incorrect step, briefly</error_location>\n"
        "<error>\n"
        "One or two sentences explaining the student's main mistake. If the solution is already correct, say it is correct.\n"
        "</error>\n"
        "<corrected_solution>\n"
        "Concise corrected reasoning.\n"
        "</corrected_solution>\n"
        "<final_answer>the final number only</final_answer>\n"
        "Answer: the final number only\n\n"
        "The final line must begin with `Answer:` and contain only the final numeric answer after it. "
        "Do not use boxed answers or the GSM8K #### marker.\n\n"
        "<problem>\n"
        f"{str(original_prompt).strip()}\n"
        "</problem>\n\n"
        "<student_previous_solution>\n"
        f"{str(candidate_solution).strip()}\n"
        "</student_previous_solution>\n"
    )


def teacher_prompt(original_prompt, candidate_solution):
    return (
        "You are a careful math teacher. Your job is to teach a small model how to correct a failed reasoning trace.\n"
        "Given the original problem and the student's previous solution, diagnose the first important error and provide a corrected solution.\n\n"
        "Rules:\n"
        "1. Do not merely state the answer; show the corrected reasoning compactly.\n"
        "2. The corrected solution must be mathematically correct.\n"
        "3. Keep the final answer easy to parse.\n\n"
        + correction_prompt(original_prompt, candidate_solution)
    )


def tag_text(text, tag):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", str(text or ""), re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def teacher_format_ok(text):
    text = str(text or "")
    lower = text.lower()
    return all(
        marker in lower
        for marker in ["<error_type>", "</error_type>", "<error_location>", "</error_location>", "<corrected_solution>", "</corrected_solution>", "<final_answer>", "</final_answer>"]
    ) and bool(re.search(r"(?:^|\n)\s*Answer:\s*\S+", text))


def cache_key(teacher_model, row, rollout):
    payload = {
        "prompt_version": PROMPT_VERSION,
        "teacher_model": teacher_model,
        "row_id": row["id"],
        "reference_answer": row["reference_answer"],
        "student_solution": rollout["response"],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def call_openai_chat(api_base, api_key, model_name, messages, args):
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": args.teacher_temperature,
        "top_p": args.teacher_top_p,
        "max_tokens": args.teacher_max_tokens,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=args.teacher_timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def candidate_rollouts(group, args):
    failed = [r for r in group.get("rollouts", []) if int(r.get("ok", 0)) == 0 and str(r.get("response", "")).strip()]
    if args.prefer_unclipped:
        unclipped = [r for r in failed if not r.get("clipped")]
        if unclipped:
            failed = unclipped
    failed.sort(key=lambda r: (int(r.get("clipped", False)), -int(r.get("format_ok", False)), int(r.get("new_tokens", 10**9))))
    return failed[: args.max_candidates_per_question]


def selected_requests(groups, args, rng, existing_keys, cache):
    eligible = []
    for group in groups:
        pass_rate = float(group.get("pass_rate", 0.0))
        if pass_rate > args.max_pass_rate:
            continue
        row = {
            "id": group["id"],
            "idx": group.get("idx"),
            "prompt": group["prompt"],
            "question": group.get("question", ""),
            "reference_answer": group["reference_answer"],
            "pass_rate": pass_rate,
        }
        for rollout in candidate_rollouts(group, args):
            key = cache_key(args.teacher_model, row, rollout)
            if key in existing_keys:
                continue
            eligible.append((pass_rate, not usable_cache_record(cache.get(key), row["reference_answer"]), row, rollout, key))
    eligible.sort(key=lambda x: (x[0], not x[1], str(x[2]["id"])))
    if args.shuffle:
        rng.shuffle(eligible)
    return eligible[: args.max_samples if args.max_samples > 0 else None]


def build_sft_record(row, rollout, key, teacher_text, args, cached):
    verdict = verify_math_response(teacher_text, row["reference_answer"])
    fmt_ok = teacher_format_ok(teacher_text)
    sft_prompt = correction_prompt(row["prompt"], rollout["response"])
    return {
        "schema": SCHEMA,
        "key": key,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt_version": PROMPT_VERSION,
        "teacher_model": args.teacher_model,
        "cached": bool(cached),
        "idx": row.get("idx"),
        "id": row["id"],
        "original_prompt": row["prompt"],
        "question": row.get("question", ""),
        "reference_answer": row["reference_answer"],
        "pass_rate": row["pass_rate"],
        "student_solution": rollout["response"],
        "student_predicted_answer": rollout.get("predicted_answer", ""),
        "student_new_tokens": rollout.get("new_tokens", 0),
        "student_clipped": bool(rollout.get("clipped", False)),
        "student_format_ok": bool(rollout.get("format_ok", False)),
        "teacher_response": teacher_text,
        "teacher_predicted_answer": verdict["predicted_answer"],
        "teacher_final_answer": extract_final_answer(teacher_text),
        "teacher_verified": bool(verdict["reward"]),
        "teacher_format_ok": fmt_ok,
        "teacher_error_type": tag_text(teacher_text, "error_type"),
        "teacher_error_location": tag_text(teacher_text, "error_location"),
        "teacher_error": tag_text(teacher_text, "error"),
        "prompt": sft_prompt,
        "response": teacher_text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--output", default="data/teacher_correction/correction_sft_train.jsonl")
    parser.add_argument("--cache", default="data/teacher_correction/teacher_cache.jsonl")
    parser.add_argument("--env-file", default=".env.teacher")
    parser.add_argument("--teacher-api-base", default="")
    parser.add_argument("--teacher-api-key-env", default="TEACHER_API_KEY")
    parser.add_argument("--teacher-model", default="")
    parser.add_argument("--teacher-max-tokens", type=int, default=4096)
    parser.add_argument("--teacher-temperature", type=float, default=0.2)
    parser.add_argument("--teacher-top-p", type=float, default=0.95)
    parser.add_argument("--teacher-timeout", type=float, default=90.0)
    parser.add_argument("--teacher-retries", type=int, default=2)
    parser.add_argument("--teacher-retry-sleep", type=float, default=2.0)
    parser.add_argument("--teacher-concurrency", type=int, default=128)
    parser.add_argument("--max-pass-rate", type=float, default=0.25)
    parser.add_argument("--max-candidates-per-question", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--prefer-unclipped", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-correct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-format", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=30)
    args = parser.parse_args()

    load_env_file(args.env_file)
    if not args.teacher_api_base:
        args.teacher_api_base = os.environ.get("TEACHER_API_BASE", "https://api.openai.com/v1")
    if not args.teacher_model:
        args.teacher_model = os.environ.get("TEACHER_MODEL", "teacher-model")
    api_key = os.environ.get(args.teacher_api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing ${args.teacher_api_key_env}; pass --env-file or export it first.")

    rng = random.Random(args.seed)
    groups = list(read_jsonl(args.rollouts))
    cache = load_cache(args.cache)
    existing_keys = load_existing_keys(args.output)
    requests = selected_requests(groups, args, rng, existing_keys, cache)
    if not requests:
        print(json.dumps({"message": "no new teacher requests", "output": args.output}, ensure_ascii=False), flush=True)
        return

    stats = {
        "requested": len(requests),
        "api_calls": 0,
        "cache_hits": 0,
        "written": 0,
        "skipped_incorrect": 0,
        "skipped_bad_format": 0,
        "failed": 0,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    def fetch_request(req):
        _, _, row, rollout, key = req
        rec = cache.get(key)
        if usable_cache_record(rec, row["reference_answer"]):
            return {
                "row": row,
                "rollout": rollout,
                "key": key,
                "cached": True,
                "teacher_text": rec.get("teacher_response", ""),
                "cache_rec": None,
                "error": None,
            }

        messages = [
            {"role": "system", "content": "You are a precise math reasoning teacher."},
            {"role": "user", "content": teacher_prompt(row["prompt"], rollout["response"])},
        ]
        last_error = None
        teacher_text = ""
        for attempt in range(max(1, args.teacher_retries)):
            try:
                teacher_text = call_openai_chat(args.teacher_api_base, api_key, args.teacher_model, messages, args)
                last_error = None
                break
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                time.sleep(args.teacher_retry_sleep * (attempt + 1))
        if last_error:
            return {
                "row": row,
                "rollout": rollout,
                "key": key,
                "cached": False,
                "teacher_text": "",
                "cache_rec": {
                    "key": key,
                    "ok": False,
                    "error": last_error[:500],
                    "row_id": row["id"],
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "error": last_error,
            }
        return {
            "row": row,
            "rollout": rollout,
            "key": key,
            "cached": False,
            "teacher_text": teacher_text,
            "cache_rec": {
                "key": key,
                "ok": True,
                "row_id": row["id"],
                "teacher_model": args.teacher_model,
                "prompt_version": PROMPT_VERSION,
                "teacher_response": teacher_text,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "error": None,
        }

    if args.teacher_concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.teacher_concurrency) as pool:
            futures = {pool.submit(fetch_request, req): n for n, req in enumerate(requests, 1)}
            iterator = ((futures[fut], fut.result()) for fut in as_completed(futures))
            for i, result in iterator:
                row = result["row"]
                rollout = result["rollout"]
                key = result["key"]
                cached = result["cached"]
                if cached:
                    stats["cache_hits"] += 1
                elif result["cache_rec"] is not None:
                    append_jsonl(args.cache, result["cache_rec"])
                    cache[key] = result["cache_rec"]
                    stats["api_calls"] += int(result["cache_rec"].get("ok", False))
                if result["error"]:
                    stats["failed"] += 1
                    print(json.dumps({"progress": f"{i}/{len(requests)}", "row_id": row["id"], "failed": result["error"][:120]}, ensure_ascii=False), flush=True)
                    continue

                sft_rec = build_sft_record(row, rollout, key, result["teacher_text"], args, cached)
                if args.require_format and not sft_rec["teacher_format_ok"]:
                    stats["skipped_bad_format"] += 1
                    status = "bad_format"
                elif args.require_correct and not sft_rec["teacher_verified"]:
                    stats["skipped_incorrect"] += 1
                    status = "incorrect"
                else:
                    append_jsonl(args.output, sft_rec)
                    stats["written"] += 1
                    status = "written"
                print(
                    json.dumps(
                        {
                            "progress": f"{i}/{len(requests)}",
                            "row_id": row["id"],
                            "status": status,
                            "teacher_verified": sft_rec["teacher_verified"],
                            "teacher_format_ok": sft_rec["teacher_format_ok"],
                            "teacher_final_answer": sft_rec["teacher_final_answer"],
                            "pass_rate": row["pass_rate"],
                            "cached": cached,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    else:
        for i, req in enumerate(requests, 1):
            result = fetch_request(req)
            row = result["row"]
            rollout = result["rollout"]
            key = result["key"]
            cached = result["cached"]
            if cached:
                stats["cache_hits"] += 1
            elif result["cache_rec"] is not None:
                append_jsonl(args.cache, result["cache_rec"])
                cache[key] = result["cache_rec"]
                stats["api_calls"] += int(result["cache_rec"].get("ok", False))
            if result["error"]:
                stats["failed"] += 1
                print(json.dumps({"progress": f"{i}/{len(requests)}", "row_id": row["id"], "failed": result["error"][:120]}, ensure_ascii=False), flush=True)
                continue

            sft_rec = build_sft_record(row, rollout, key, result["teacher_text"], args, cached)
            if args.require_format and not sft_rec["teacher_format_ok"]:
                stats["skipped_bad_format"] += 1
                status = "bad_format"
            elif args.require_correct and not sft_rec["teacher_verified"]:
                stats["skipped_incorrect"] += 1
                status = "incorrect"
            else:
                append_jsonl(args.output, sft_rec)
                stats["written"] += 1
                status = "written"
            print(
                json.dumps(
                    {
                        "progress": f"{i}/{len(requests)}",
                        "row_id": row["id"],
                        "status": status,
                        "teacher_verified": sft_rec["teacher_verified"],
                        "teacher_format_ok": sft_rec["teacher_format_ok"],
                        "teacher_final_answer": sft_rec["teacher_final_answer"],
                        "pass_rate": row["pass_rate"],
                        "cached": cached,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary_path = str(Path(args.output).with_suffix(".summary.json"))
    summary = {
        "schema": "teacher_correction_build_summary_v1",
        "rollouts": args.rollouts,
        "output": args.output,
        "cache": args.cache,
        "teacher_model": args.teacher_model,
        "teacher_concurrency": args.teacher_concurrency,
        "max_pass_rate": args.max_pass_rate,
        **stats,
    }
    Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
