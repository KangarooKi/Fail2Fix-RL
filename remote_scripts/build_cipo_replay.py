import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verifier_math import verify_math_response


def read_jsonl(path, limit=None, offset=0):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < offset or not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("id", str(idx))
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chat_prompt(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": str(prompt).strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )


def correction_prompt(question_prompt, previous_response, anchor_reward):
    status = "correct" if int(anchor_reward) == 1 else "incorrect"
    return (
        "You are revising a previous attempt on a grade-school math problem.\n\n"
        f"Original problem:\n{question_prompt.strip()}\n\n"
        f"Previous attempt, judged {status} by an automatic verifier:\n"
        f"{previous_response.strip()}\n\n"
        "Re-solve the original problem. If the previous attempt is wrong, identify the key mistake silently and fix it. "
        "If it is already correct, reproduce a concise correct solution. "
        "The last line must be exactly: Answer: <final number>"
    )


def generate_rollouts(model, tokenizer, prompts, args):
    device = next(model.parameters()).device
    texts = [chat_prompt(tokenizer, p) for p in prompts]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_prompt_length,
    ).to(device)
    input_len = int(encoded["input_ids"].shape[-1])
    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=args.rollouts_per_prompt,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    responses = []
    for seq in output_ids:
        new_ids = seq[input_len:]
        text = tokenizer.decode(new_ids, skip_special_tokens=False)
        clipped = bool(
            new_ids.numel() >= args.max_new_tokens
            and (tokenizer.eos_token_id is None or int(new_ids[-1]) != tokenizer.eos_token_id)
        )
        responses.append({"response": text, "new_tokens": int(new_ids.numel()), "clipped": clipped})
    grouped = []
    n = args.rollouts_per_prompt
    for i in range(0, len(responses), n):
        grouped.append(responses[i : i + n])
    return grouped


def select_anchor_records(source_rows, rollout_groups, args):
    prompt_summaries = []
    failed = []
    correct = []
    all_candidates = []

    for row, group in zip(source_rows, rollout_groups):
        scored = []
        for rollout_idx, sample in enumerate(group):
            verdict = verify_math_response(sample["response"], row["reference_answer"])
            reward = int(verdict["reward"])
            rec = {
                "source_id": row.get("id"),
                "question": row.get("question", ""),
                "original_prompt": row["prompt"],
                "reference_answer": row["reference_answer"],
                "anchor_response": sample["response"],
                "anchor_reward": reward,
                "anchor_predicted_answer": verdict["predicted_answer"],
                "anchor_new_tokens": sample["new_tokens"],
                "anchor_clipped": sample["clipped"],
                "rollout_idx": rollout_idx,
            }
            scored.append(rec)
            all_candidates.append(rec)

        pass_rate = sum(r["anchor_reward"] for r in scored) / len(scored) if scored else 0.0
        for rec in scored:
            rec["empirical_pass_rate"] = pass_rate
        prompt_summaries.append(
            {
                "id": row.get("id"),
                "reference_answer": row["reference_answer"],
                "pass_rate": pass_rate,
                "num_rollouts": len(scored),
                "num_correct": sum(r["anchor_reward"] for r in scored),
            }
        )
        if args.min_pass_rate <= pass_rate <= args.max_pass_rate:
            failed.extend([r for r in scored if r["anchor_reward"] == 0])
            correct.extend([r for r in scored if r["anchor_reward"] == 1])

    if not failed:
        failed = [r for r in all_candidates if r["anchor_reward"] == 0]
    if not correct:
        correct = [r for r in all_candidates if r["anchor_reward"] == 1]

    random.shuffle(failed)
    random.shuffle(correct)
    target_total = args.target_correction_rows or len(source_rows)
    target_correct = min(len(correct), int(round(target_total * args.correct_anchor_ratio)))
    target_failed = min(len(failed), max(0, target_total - target_correct))
    anchors = failed[:target_failed] + correct[:target_correct]
    random.shuffle(anchors)
    return anchors, prompt_summaries


def build_train_rows(source_rows, anchors, args):
    rows = []
    for row in source_rows:
        rows.append(
            {
                "id": f"base::{row.get('id')}",
                "source_id": row.get("id"),
                "stream": "base",
                "prompt": row["prompt"],
                "question": row.get("question", ""),
                "reference_answer": row["reference_answer"],
                "anchor_reward": -1,
                "empirical_pass_rate": None,
            }
        )

    for i, anchor in enumerate(anchors):
        rows.append(
            {
                "id": f"cipo::{anchor['source_id']}::{anchor['rollout_idx']}::{i}",
                "source_id": anchor["source_id"],
                "stream": "correction",
                "prompt": correction_prompt(anchor["original_prompt"], anchor["anchor_response"], anchor["anchor_reward"]),
                "question": anchor.get("question", ""),
                "reference_answer": anchor["reference_answer"],
                "anchor_reward": anchor["anchor_reward"],
                "anchor_predicted_answer": anchor["anchor_predicted_answer"],
                "anchor_new_tokens": anchor["anchor_new_tokens"],
                "anchor_clipped": anchor["anchor_clipped"],
                "empirical_pass_rate": anchor["empirical_pass_rate"],
            }
        )
    random.shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/models")
    parser.add_argument("--data", default="/root/autodl-tmp/learning_from_failure_exp/data/gsm8k_grpo/train.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollouts-per-prompt", type=int, default=8)
    parser.add_argument("--target-correction-rows", type=int, default=0)
    parser.add_argument("--correct-anchor-ratio", type=float, default=0.2)
    parser.add_argument("--min-pass-rate", type=float, default=0.125)
    parser.add_argument("--max-pass-rate", type=float, default=0.875)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()
    if args.target_correction_rows <= 0:
        args.target_correction_rows = args.limit

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = read_jsonl(args.data, args.limit if args.limit > 0 else None, args.offset)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    all_groups = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        groups = generate_rollouts(model, tokenizer, [r["prompt"] for r in batch], args)
        all_groups.extend(groups)
        done = min(start + len(batch), len(rows))
        print(json.dumps({"build_cipo_replay": {"done": done, "total": len(rows)}}, ensure_ascii=False), flush=True)

    anchors, summaries = select_anchor_records(rows, all_groups, args)
    train_rows = build_train_rows(rows, anchors, args)
    write_jsonl(args.output, train_rows)

    num_base = sum(1 for r in train_rows if r["stream"] == "base")
    num_corr = sum(1 for r in train_rows if r["stream"] == "correction")
    num_corr_from_failed = sum(1 for r in train_rows if r["stream"] == "correction" and int(r["anchor_reward"]) == 0)
    num_corr_from_correct = sum(1 for r in train_rows if r["stream"] == "correction" and int(r["anchor_reward"]) == 1)
    pass_rates = [s["pass_rate"] for s in summaries]
    report = {
        "model": args.model,
        "data": args.data,
        "source_rows": len(rows),
        "rollouts_per_prompt": args.rollouts_per_prompt,
        "train_rows": len(train_rows),
        "base_rows": num_base,
        "correction_rows": num_corr,
        "correction_from_failed": num_corr_from_failed,
        "correction_from_correct": num_corr_from_correct,
        "min_pass_rate": args.min_pass_rate,
        "max_pass_rate": args.max_pass_rate,
        "avg_empirical_pass_rate": sum(pass_rates) / len(pass_rates) if pass_rates else 0.0,
        "num_all_wrong_prompts": sum(1 for p in pass_rates if p == 0.0),
        "num_all_correct_prompts": sum(1 for p in pass_rates if p == 1.0),
        "num_mixed_prompts": sum(1 for p in pass_rates if 0.0 < p < 1.0),
        "prompt_summaries": summaries[:200],
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cipo_replay_report": report}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
