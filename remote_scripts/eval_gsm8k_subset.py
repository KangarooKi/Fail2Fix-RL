import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verifier_math import verify_math_response


def load_rows(path, limit=None, offset=0):
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
                    "reference_answer": row["reference_answer"],
                    "source": row.get("source", "gsm8k"),
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def safe_name(text):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return text.strip("_")[:120] or "model"


def prompt_text(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": str(prompt).strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="/root/autodl-tmp/learning_from_failure_exp/data/gsm8k_grpo/test.jsonl")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/learning_from_failure_exp/reports/eval_gsm8k")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.data, args.limit if args.limit > 0 else None, args.offset)
    if not rows:
        raise ValueError(f"No rows loaded from {args.data}")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = safe_name(Path(args.model).name)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    pred_path = out_dir / f"{tag}_gsm8k_eval_n{len(rows)}_o{args.offset}_{stamp}.jsonl"
    summary_path = out_dir / f"{tag}_gsm8k_eval_n{len(rows)}_o{args.offset}_{stamp}.json"

    correct = 0
    total_new_tokens = 0
    clipped = 0
    started = time.time()
    with pred_path.open("w", encoding="utf-8") as f:
        for j, row in enumerate(rows, 1):
            text = prompt_text(tokenizer, row["prompt"])
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_prompt_length,
            ).to(device)
            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if args.temperature > 0:
                gen_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
            else:
                gen_kwargs.update({"do_sample": False})

            with torch.inference_mode():
                output_ids = model.generate(**inputs, **gen_kwargs)
            input_len = int(inputs["input_ids"].shape[-1])
            new_ids = output_ids[0][input_len:]
            response = tokenizer.decode(new_ids, skip_special_tokens=False)
            new_token_count = int(new_ids.shape[-1])
            total_new_tokens += new_token_count
            is_clipped = new_token_count >= args.max_new_tokens and (
                tokenizer.eos_token_id is None or int(new_ids[-1]) != tokenizer.eos_token_id
            )
            clipped += int(is_clipped)

            verdict = verify_math_response(response, row["reference_answer"])
            ok = int(verdict["reward"])
            correct += ok
            rec = {
                "idx": row["idx"],
                "id": row["id"],
                "ok": ok,
                "predicted_answer": verdict["predicted_answer"],
                "reference_answer": row["reference_answer"],
                "new_tokens": new_token_count,
                "clipped": bool(is_clipped),
                "response": response,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(
                json.dumps(
                    {
                        "progress": f"{j}/{len(rows)}",
                        "running_acc": correct / j,
                        "ok": ok,
                        "new_tokens": new_token_count,
                        "clipped": bool(is_clipped),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if device == "cuda":
                torch.cuda.empty_cache()

    elapsed = time.time() - started
    summary = {
        "model": args.model,
        "data": args.data,
        "limit": len(rows),
        "offset": args.offset,
        "accuracy": correct / len(rows),
        "correct": correct,
        "total": len(rows),
        "avg_new_tokens": total_new_tokens / len(rows),
        "clipped": clipped,
        "clipped_ratio": clipped / len(rows),
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "device": device,
        "elapsed_sec": elapsed,
        "predictions_path": str(pred_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
