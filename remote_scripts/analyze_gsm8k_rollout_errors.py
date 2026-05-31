import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verifier_math import verify_math_response


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_data_by_id(path):
    rows = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("idx", i)
            row.setdefault("id", row.get("id", str(i)))
            rows[row["id"]] = row
            rows[str(row.get("idx", i))] = row
    return rows


def row_key(row):
    return row.get("id") or str(row.get("idx"))


def choose_hard_rows(data_by_id, eval_predictions, base_predictions=None, num_questions=4, seed=29):
    base_ok = {}
    if base_predictions:
        for row in base_predictions:
            base_ok[row_key(row)] = int(row.get("ok", 0))
            base_ok[str(row.get("idx"))] = int(row.get("ok", 0))

    candidates = []
    fallback = []
    for pred in eval_predictions:
        key = row_key(pred)
        data_row = data_by_id.get(key) or data_by_id.get(str(pred.get("idx")))
        if not data_row:
            continue
        if int(pred.get("ok", 0)) == 0:
            fallback.append(data_row)
            if not base_predictions or base_ok.get(key, base_ok.get(str(pred.get("idx")), 0)) == 0:
                candidates.append(data_row)

    pool = candidates or fallback
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:num_questions]


def prompt_text(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": str(prompt).strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )


def short_tail(text, max_chars=420):
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    tail = "\n".join(lines[-4:]) if lines else str(text or "")
    tail = re.sub(r"\s+", " ", tail).strip()
    if len(tail) > max_chars:
        return tail[-max_chars:]
    return tail


def classify_error(response, predicted, clipped, ok):
    if ok:
        return "correct"
    if clipped:
        return "truncated_or_runaway"
    pred = str(predicted or "").strip()
    if not pred:
        return "no_final_answer_extracted"
    has_answer_marker = bool(re.search(r"(answer|答案|final answer|\\boxed|<answer>)", str(response or ""), re.IGNORECASE))
    has_number = bool(re.search(r"-?\d", pred))
    if not has_answer_marker:
        return "missing_required_answer_format"
    if has_number:
        return "wrong_numeric_reasoning"
    return "non_numeric_or_unparseable_final"


def generate_rollouts(model, tokenizer, row, args, device):
    text = prompt_text(tokenizer, row["prompt"])
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_prompt_length,
    ).to(device)
    records = []
    remaining = args.num_rollouts
    while remaining > 0:
        batch = min(args.rollout_batch_size, remaining)
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
        input_len = int(inputs["input_ids"].shape[-1])
        for seq in output_ids:
            new_ids = seq[input_len:]
            response = tokenizer.decode(new_ids, skip_special_tokens=False)
            new_token_count = int(new_ids.shape[-1])
            clipped = new_token_count >= args.max_new_tokens and (
                tokenizer.eos_token_id is None or int(new_ids[-1]) != tokenizer.eos_token_id
            )
            verdict = verify_math_response(response, row["reference_answer"])
            ok = int(verdict["reward"])
            predicted = verdict["predicted_answer"]
            records.append(
                {
                    "ok": ok,
                    "predicted_answer": predicted,
                    "reference_answer": row["reference_answer"],
                    "new_tokens": new_token_count,
                    "clipped": bool(clipped),
                    "error_type": classify_error(response, predicted, clipped, ok),
                    "tail": short_tail(response),
                    "response": response,
                }
            )
        remaining -= batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def write_markdown(path, summary):
    lines = []
    lines.append("# GSM8K 16-Rollout Error Analysis")
    lines.append("")
    lines.append(f"- model: `{summary['model']}`")
    lines.append(f"- generated_at: `{summary['generated_at']}`")
    lines.append(f"- sampling: `G={summary['num_rollouts']}`, `temperature={summary['temperature']}`, `top_p={summary['top_p']}`, `max_new_tokens={summary['max_new_tokens']}`")
    lines.append("")
    for q in summary["questions"]:
        lines.append(f"## {q['id']}")
        lines.append("")
        lines.append(f"**Reference answer:** `{q['reference_answer']}`")
        lines.append("")
        lines.append(q["question"])
        lines.append("")
        lines.append(
            f"- correct: `{q['correct']}/{q['total']}`"
            f"; clipped: `{q['clipped']}/{q['total']}`"
            f"; avg_new_tokens: `{q['avg_new_tokens']:.1f}`"
        )
        lines.append(f"- answer histogram: `{json.dumps(q['answer_histogram'], ensure_ascii=False)}`")
        lines.append(f"- error types: `{json.dumps(q['error_types'], ensure_ascii=False)}`")
        lines.append("")
        lines.append("| # | ok | pred | tokens | clipped | error_type | tail |")
        lines.append("|---:|---:|---|---:|---:|---|---|")
        for i, r in enumerate(q["rollouts"], 1):
            tail = r["tail"].replace("|", "\\|")
            pred = str(r["predicted_answer"]).replace("|", "\\|")
            lines.append(f"| {i} | {r['ok']} | `{pred}` | {r['new_tokens']} | {int(r['clipped'])} | {r['error_type']} | {tail} |")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="/root/autodl-tmp/learning_from_failure_exp/data/gsm8k_grpo/test.jsonl")
    parser.add_argument("--eval-predictions", required=True)
    parser.add_argument("--base-predictions", default=None)
    parser.add_argument("--output-dir", default="/root/autodl-tmp/learning_from_failure_exp/reports/rollout_error_analysis")
    parser.add_argument("--num-questions", type=int, default=4)
    parser.add_argument("--num-rollouts", type=int, default=16)
    parser.add_argument("--rollout-batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_by_id = load_data_by_id(args.data)
    eval_predictions = load_jsonl(args.eval_predictions)
    base_predictions = load_jsonl(args.base_predictions) if args.base_predictions else None
    hard_rows = choose_hard_rows(data_by_id, eval_predictions, base_predictions, args.num_questions, args.seed)
    if not hard_rows:
        raise ValueError("No hard rows selected.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True).to(device)
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"gsm8k_rollout_errors_{stamp}.json"
    md_path = out_dir / f"gsm8k_rollout_errors_{stamp}.md"

    questions = []
    for row in hard_rows:
        rollouts = generate_rollouts(model, tokenizer, row, args, device)
        correct = sum(r["ok"] for r in rollouts)
        clipped = sum(int(r["clipped"]) for r in rollouts)
        answer_hist = Counter(str(r["predicted_answer"]) for r in rollouts)
        error_types = Counter(r["error_type"] for r in rollouts)
        q = {
            "id": row.get("id", str(row.get("idx"))),
            "idx": row.get("idx"),
            "question": row.get("question", row["prompt"].split("\n\n")[-1]),
            "reference_answer": row["reference_answer"],
            "solution": row.get("solution", ""),
            "correct": correct,
            "total": len(rollouts),
            "clipped": clipped,
            "avg_new_tokens": sum(r["new_tokens"] for r in rollouts) / len(rollouts),
            "answer_histogram": dict(answer_hist.most_common()),
            "error_types": dict(error_types.most_common()),
            "rollouts": rollouts,
        }
        questions.append(q)
        print(json.dumps({k: q[k] for k in ("id", "correct", "total", "clipped", "answer_histogram", "error_types")}, ensure_ascii=False), flush=True)

    summary = {
        "model": args.model,
        "data": args.data,
        "eval_predictions": args.eval_predictions,
        "base_predictions": args.base_predictions,
        "generated_at": stamp,
        "num_questions": len(questions),
        "num_rollouts": args.num_rollouts,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "questions": questions,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, summary)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
