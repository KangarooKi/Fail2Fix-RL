import argparse
import json
import random
from collections import Counter
from pathlib import Path

from modelscope.msdatasets import MsDataset


def make_record(row, index, split):
    reward_model = row.get("reward_model") or {}
    answer = str(reward_model.get("ground_truth") or row.get("solution") or "").strip()
    source_prompt = row.get("source_prompt") or []
    if source_prompt and isinstance(source_prompt, list):
        prompt = str(source_prompt[0].get("content") or "").strip()
    else:
        prompt = (
            "Solve the following math problem step by step. The last line of your response "
            "should be of the form Answer: $Answer.\n\n"
            + str(row.get("prompt") or "").strip()
        )
    return {
        "id": f"dapo/{row.get('extra_info', {}).get('index', index)}",
        "prompt": prompt,
        "reference_answer": answer,
        "source": "open-r1/DAPO-Math-17k-Processed",
        "source_type": "math",
        "data_source": row.get("data_source"),
        "ability": row.get("ability"),
        "reward_style": reward_model.get("style"),
        "split": split,
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/root/autodl-tmp/learning_from_failure_exp/data/dapo_math17k_grpo")
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260529)
    args = parser.parse_args()

    ds = list(MsDataset.load("open-r1/DAPO-Math-17k-Processed", split="train"))
    rng = random.Random(args.seed)
    order = list(range(len(ds)))
    rng.shuffle(order)
    eval_ids = set(order[: args.eval_size])

    train_rows = []
    eval_rows = []
    for i, row in enumerate(ds):
        split = "eval" if i in eval_ids else "train"
        record = make_record(row, i, split)
        if not record["prompt"] or not record["reference_answer"]:
            continue
        (eval_rows if split == "eval" else train_rows).append(record)

    out = Path(args.output_dir)
    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "eval.jsonl", eval_rows)
    report = {
        "dataset": "open-r1/DAPO-Math-17k-Processed",
        "seed": args.seed,
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "train_sources": dict(Counter(r["data_source"] for r in train_rows)),
        "eval_sources": dict(Counter(r["data_source"] for r in eval_rows)),
    }
    (out / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
