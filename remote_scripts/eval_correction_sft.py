import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verifier_math import verify_math_response


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


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


def chat_text(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": str(prompt).strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )


def format_ok(text):
    lower = str(text or "").lower()
    return "<final_answer>" in lower and bool(re.search(r"(?:^|\n)\s*Answer:\s*\S+", str(text or "")))


def select_cases(rollout_path, limit, max_pass_rate, prefer_unclipped=True):
    cases = []
    for group in read_jsonl(rollout_path):
        if float(group.get("pass_rate", 0.0)) > max_pass_rate:
            continue
        failed = [r for r in group.get("rollouts", []) if int(r.get("ok", 0)) == 0 and str(r.get("response", "")).strip()]
        if prefer_unclipped:
            unclipped = [r for r in failed if not r.get("clipped")]
            if unclipped:
                failed = unclipped
        if not failed:
            continue
        failed.sort(key=lambda r: (int(r.get("clipped", False)), int(r.get("new_tokens", 10**9))))
        rollout = failed[0]
        cases.append(
            {
                "id": group["id"],
                "idx": group.get("idx"),
                "prompt": group["prompt"],
                "reference_answer": group["reference_answer"],
                "pass_rate": group.get("pass_rate", 0.0),
                "student_solution": rollout["response"],
                "student_predicted_answer": rollout.get("predicted_answer", ""),
                "student_new_tokens": rollout.get("new_tokens", 0),
                "student_clipped": bool(rollout.get("clipped", False)),
            }
        )
        if limit > 0 and len(cases) >= limit:
            break
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--output-dir", default="/root/autodl-tmp/learning_from_failure_exp/reports/correction_eval")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--max-pass-rate", type=float, default=0.25)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    cases = select_cases(args.rollouts, args.limit, args.max_pass_rate)
    if not cases:
        raise ValueError("No correction eval cases selected.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True).to(device)
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    pred_path = out_dir / f"correction_eval_{Path(args.model).name}_n{len(cases)}_{stamp}.jsonl"
    summary_path = out_dir / f"correction_eval_{Path(args.model).name}_n{len(cases)}_{stamp}.json"

    correct = 0
    fmt = 0
    clipped_count = 0
    total_new_tokens = 0
    started = time.time()
    with pred_path.open("w", encoding="utf-8") as f:
        for i, case in enumerate(cases, 1):
            prompt = correction_prompt(case["prompt"], case["student_solution"])
            text = chat_text(tokenizer, prompt)
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_prompt_length,
                add_special_tokens=False,
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
            response = tokenizer.decode(new_ids, skip_special_tokens=True)
            new_tokens = int(new_ids.shape[-1])
            clipped = new_tokens >= args.max_new_tokens and (
                tokenizer.eos_token_id is None or int(new_ids[-1]) != tokenizer.eos_token_id
            )
            verdict = verify_math_response(response, case["reference_answer"])
            ok = int(verdict["reward"])
            rec = {
                **case,
                "ok": ok,
                "predicted_answer": verdict["predicted_answer"],
                "new_tokens": new_tokens,
                "clipped": bool(clipped),
                "format_ok": format_ok(response),
                "response": response,
            }
            correct += ok
            fmt += int(rec["format_ok"])
            clipped_count += int(clipped)
            total_new_tokens += new_tokens
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(
                json.dumps(
                    {
                        "progress": f"{i}/{len(cases)}",
                        "running_acc": correct / i,
                        "ok": ok,
                        "format_ok": rec["format_ok"],
                        "new_tokens": new_tokens,
                        "clipped": bool(clipped),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "model": args.model,
        "rollouts": args.rollouts,
        "total": len(cases),
        "correct": correct,
        "accuracy": correct / len(cases),
        "format_ok": fmt,
        "format_ok_ratio": fmt / len(cases),
        "clipped": clipped_count,
        "clipped_ratio": clipped_count / len(cases),
        "avg_new_tokens": total_new_tokens / len(cases),
        "max_pass_rate": args.max_pass_rate,
        "max_new_tokens": args.max_new_tokens,
        "predictions_path": str(pred_path),
        "elapsed_sec": time.time() - started,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
