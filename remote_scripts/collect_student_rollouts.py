import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verifier_math import verify_math_response


SCHEMA = "student_rollout_group_v1"


def read_jsonl(path, limit=None, offset=0):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset or not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                {
                    "idx": row.get("idx", i),
                    "id": row.get("id", str(i)),
                    "prompt": row["prompt"],
                    "question": row.get("question", ""),
                    "reference_answer": row["reference_answer"],
                    "source": row.get("source", "gsm8k"),
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_done_ids(path):
    done = set()
    path = Path(path)
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("schema") == SCHEMA and rec.get("id") is not None:
                done.add(str(rec["id"]))
    return done


def chat_text(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": str(prompt).strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )


def format_ok(text):
    text = str(text or "")
    return bool(("answer:" in text.lower()) or ("<final_answer>" in text.lower()) or ("\\boxed" in text))


def trim_right_pad(ids, pad_token_id):
    ids = list(ids)
    if pad_token_id is None:
        return ids
    while ids and int(ids[-1]) == int(pad_token_id):
        ids.pop()
    return ids


def generate_one_group(model, tokenizer, row, args, device):
    return generate_batch_groups(model, tokenizer, [row], args, device)[0]


def generate_batch_groups(model, tokenizer, rows, args, device):
    prompt_texts = [chat_text(tokenizer, row["prompt"]) for row in rows]
    inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_prompt_length,
        add_special_tokens=False,
    ).to(device)
    input_len = int(inputs["input_ids"].shape[-1])
    groups = [[] for _ in rows]
    remaining = args.group_size
    while remaining > 0:
        batch = min(args.generation_batch_size, remaining)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=batch,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for row_idx, row in enumerate(rows):
            for local_idx in range(batch):
                out_idx = row_idx * batch + local_idx
                new_ids = trim_right_pad(output_ids[out_idx][input_len:].detach().cpu().tolist(), tokenizer.pad_token_id)
                response = tokenizer.decode(new_ids, skip_special_tokens=True)
                new_tokens = len(new_ids)
                clipped = new_tokens >= args.max_new_tokens and (
                    tokenizer.eos_token_id is None or (new_ids and int(new_ids[-1]) != tokenizer.eos_token_id)
                )
                verdict = verify_math_response(response, row["reference_answer"])
                groups[row_idx].append(
                    {
                        "sample_idx": len(groups[row_idx]),
                        "ok": int(verdict["reward"]),
                        "predicted_answer": verdict["predicted_answer"],
                        "reference_final_answer": verdict["reference_final_answer"],
                        "new_tokens": new_tokens,
                        "clipped": bool(clipped),
                        "format_ok": format_ok(response),
                        "response": response,
                    }
                )
        remaining -= batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/models")
    parser.add_argument("--data", default="/root/autodl-tmp/learning_from_failure_exp/data/gsm8k_grpo/train.jsonl")
    parser.add_argument("--output", default="/root/autodl-tmp/learning_from_failure_exp/data/teacher_correction/student_rollouts_train.jsonl")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--prompt-batch-size", type=int, default=1)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = read_jsonl(args.data, args.limit if args.limit > 0 else None, args.offset)
    if not rows:
        raise ValueError(f"No rows loaded from {args.data}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and out_path.exists():
        out_path.unlink()
    done_ids = load_done_ids(out_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True).to(device)
    model.eval()

    started = time.time()
    processed = 0
    with out_path.open("a", encoding="utf-8") as f:
        pending = [(j, row) for j, row in enumerate(rows, 1) if str(row["id"]) not in done_ids]
        for start in range(0, len(pending), max(1, args.prompt_batch_size)):
            chunk = pending[start : start + max(1, args.prompt_batch_size)]
            chunk_rows = [row for _, row in chunk]
            chunk_groups = generate_batch_groups(model, tokenizer, chunk_rows, args, device)
            for (j, row), rollouts in zip(chunk, chunk_groups):
                correct = sum(r["ok"] for r in rollouts)
                clipped = sum(int(r["clipped"]) for r in rollouts)
                rec = {
                    "schema": SCHEMA,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model": args.model,
                    "data": args.data,
                    "idx": row["idx"],
                    "id": row["id"],
                    "source": row["source"],
                    "prompt": row["prompt"],
                    "question": row["question"],
                    "reference_answer": row["reference_answer"],
                    "group_size": len(rollouts),
                    "pass_rate": correct / max(1, len(rollouts)),
                    "correct_count": correct,
                    "clipped_count": clipped,
                    "avg_new_tokens": sum(r["new_tokens"] for r in rollouts) / max(1, len(rollouts)),
                    "max_prompt_length": args.max_prompt_length,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "prompt_batch_size": args.prompt_batch_size,
                    "rollouts": rollouts,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                processed += 1
                print(
                    json.dumps(
                        {
                            "progress": f"{j}/{len(rows)}",
                            "written": processed,
                            "row_id": row["id"],
                            "pass_rate": rec["pass_rate"],
                            "correct": correct,
                            "clipped": clipped,
                            "avg_new_tokens": round(rec["avg_new_tokens"], 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    summary = {
        "schema": "student_rollout_collection_summary_v1",
        "output": str(out_path),
        "model": args.model,
        "data": args.data,
        "limit": len(rows),
        "offset": args.offset,
        "new_rows_written": processed,
        "group_size": args.group_size,
        "prompt_batch_size": args.prompt_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "elapsed_sec": time.time() - started,
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
