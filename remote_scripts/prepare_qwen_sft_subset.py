import argparse
import json
from pathlib import Path


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def clean_text(value):
    return (value or "").strip()


def keep(row, max_prompt_chars, max_response_chars):
    prompt = clean_text(row.get("prompt"))
    response = clean_text(row.get("response") or row.get("distilled_answer"))
    reference = clean_text(row.get("reference_answer"))
    if not prompt or not response or not reference:
        return False
    if len(prompt) > max_prompt_chars or len(response) > max_response_chars:
        return False
    return True


def write_subset(input_path, output_path, limit, max_prompt_chars, max_response_chars):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped = 0
    with output.open("w", encoding="utf-8") as out:
        for row in iter_jsonl(input_path):
            if not keep(row, max_prompt_chars, max_response_chars):
                skipped += 1
                continue
            item = {
                "id": row.get("id"),
                "prompt": clean_text(row.get("prompt")),
                "response": clean_text(row.get("response") or row.get("distilled_answer")),
                "reference_answer": clean_text(row.get("reference_answer")),
                "source": row.get("source"),
                "source_type": row.get("source_type"),
            }
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
            if count >= limit:
                break
    return {"input": input_path, "output": output_path, "count": count, "skipped_before_limit": skipped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-input", default="/root/autodl-tmp/learning_from_failure_exp/data/splits/sft_train.jsonl")
    ap.add_argument("--eval-input", default="/root/autodl-tmp/learning_from_failure_exp/data/splits/eval_holdout.jsonl")
    ap.add_argument("--sft-output", default="/root/autodl-tmp/learning_from_failure_exp/data/qwen25_05b_sft_train_2k.jsonl")
    ap.add_argument("--eval-output", default="/root/autodl-tmp/learning_from_failure_exp/data/qwen25_05b_eval_100.jsonl")
    ap.add_argument("--train-limit", type=int, default=2000)
    ap.add_argument("--eval-limit", type=int, default=100)
    ap.add_argument("--max-prompt-chars", type=int, default=3000)
    ap.add_argument("--max-response-chars", type=int, default=9000)
    args = ap.parse_args()

    train = write_subset(
        args.sft_input,
        args.sft_output,
        args.train_limit,
        args.max_prompt_chars,
        args.max_response_chars,
    )
    eval_report = write_subset(
        args.eval_input,
        args.eval_output,
        args.eval_limit,
        args.max_prompt_chars,
        args.max_response_chars,
    )
    report = {"train": train, "eval": eval_report}
    report_path = Path(args.sft_output).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
