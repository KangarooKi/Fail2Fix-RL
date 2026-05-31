import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from modelscope.msdatasets import MsDataset


ASY_RE = re.compile(r"\[asy\].*?\[/asy\]", re.DOTALL | re.IGNORECASE)
BOX_RE = re.compile(r"\\boxed\s*\{", re.DOTALL)


def clean_solution(solution, answer):
    text = ASY_RE.sub("", str(solution or "")).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if not BOX_RE.search(text):
        text = text.rstrip() + f"\n\nTherefore, the final answer is \\boxed{{{answer}}}."
    return text.strip()


def row_to_record(row, idx, split):
    answer = str(row.get("answer", "")).strip()
    return {
        "id": f"math500/{row.get('unique_id', idx)}",
        "prompt": str(row["problem"]).strip(),
        "response": clean_solution(row.get("solution", ""), answer),
        "reference_answer": answer,
        "source": "AI-ModelScope/MATH-500",
        "source_type": "math",
        "subject": row.get("subject", ""),
        "level": row.get("level", None),
        "unique_id": row.get("unique_id", str(idx)),
        "split": split,
    }


def stratified_split(rows, sft_n, eval_n, rl_n, seed):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[(row.get("subject", ""), row.get("level", None))].append((idx, row))

    buckets = {"sft": [], "eval": [], "rl": []}
    targets = {"sft": sft_n, "eval": eval_n, "rl": rl_n}
    order = ["rl", "eval", "sft"]

    shuffled_groups = list(groups.values())
    rng.shuffle(shuffled_groups)
    for group in shuffled_groups:
        rng.shuffle(group)
        for item in group:
            deficits = {name: targets[name] - len(buckets[name]) for name in targets}
            candidates = [name for name in order if deficits[name] > 0]
            if not candidates:
                raise RuntimeError("Too many rows for requested split sizes.")
            chosen = max(candidates, key=lambda name: deficits[name])
            buckets[chosen].append(item)

    return buckets


def write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def summarize(records):
    return {
        "count": len(records),
        "subjects": dict(Counter(r["subject"] for r in records)),
        "levels": dict(Counter(str(r["level"]) for r in records)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/root/autodl-tmp/learning_from_failure_exp/data/math500_sft_rl")
    parser.add_argument("--sft-size", type=int, default=400)
    parser.add_argument("--eval-size", type=int, default=50)
    parser.add_argument("--rl-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260529)
    args = parser.parse_args()

    ds = MsDataset.load("AI-ModelScope/MATH-500", split="test")
    rows = list(ds)
    total = args.sft_size + args.eval_size + args.rl_size
    if total > len(rows):
        raise ValueError(f"Requested {total} rows but dataset only has {len(rows)}.")

    buckets = stratified_split(rows, args.sft_size, args.eval_size, args.rl_size, args.seed)
    out_dir = Path(args.output_dir)
    split_records = {}
    for split, items in buckets.items():
        records = [row_to_record(row, idx, split) for idx, row in items]
        split_records[split] = records
        filename = {"sft": "sft_train.jsonl", "eval": "eval_holdout.jsonl", "rl": "rl_pool.jsonl"}[split]
        write_jsonl(out_dir / filename, records)

    report = {
        "dataset": "AI-ModelScope/MATH-500",
        "seed": args.seed,
        "sft_size": args.sft_size,
        "eval_size": args.eval_size,
        "rl_size": args.rl_size,
        "splits": {name: summarize(records) for name, records in split_records.items()},
    }
    (out_dir / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
