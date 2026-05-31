import argparse
import json
import re
from pathlib import Path

from modelscope.msdatasets import MsDataset


FINAL_RE = re.compile(r"####\s*(.+?)\s*$", re.DOTALL)


def extract_final(answer):
    text = str(answer or "").strip()
    match = FINAL_RE.search(text)
    if match:
        return match.group(1).strip().replace(",", "")
    return text.splitlines()[-1].strip().replace(",", "")


def make_prompt(question):
    return (
        "Solve the following grade-school math problem. Keep the reasoning concise. "
        "The last line must be exactly: Answer: <final number>\n\n"
        f"{str(question).strip()}"
    )


def convert(row, split, index):
    return {
        "id": f"gsm8k/{split}/{index}",
        "prompt": make_prompt(row["question"]),
        "question": row["question"],
        "reference_answer": extract_final(row["answer"]),
        "solution": row["answer"],
        "source": "AI-ModelScope/gsm8k",
        "source_type": "math",
        "split": split,
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/gsm8k_grpo")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    train_ds = MsDataset.load("AI-ModelScope/gsm8k", split="train")
    test_ds = MsDataset.load("AI-ModelScope/gsm8k", split="test")
    train_rows = [convert(row, "train", i) for i, row in enumerate(train_ds)]
    test_rows = [convert(row, "test", i) for i, row in enumerate(test_ds)]
    if args.train_limit > 0:
        train_rows = train_rows[: args.train_limit]
    if args.test_limit > 0:
        test_rows = test_rows[: args.test_limit]
    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "test.jsonl", test_rows)
    report = {
        "dataset": "AI-ModelScope/gsm8k",
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "output_dir": str(out),
        "example": train_rows[0],
    }
    (out / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
