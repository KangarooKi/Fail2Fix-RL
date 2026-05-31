import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verifier_math import verify_math_response

CORRECTION_TEMPLATE = """{original_prompt}

Below is a candidate solution from a large language model (correctness unknown):
<candidate_solution>
{candidate_solution}
</candidate_solution>

Please refer to this solution and provide your solution. Put your final answer in \\boxed{{...}} when possible."""

BASE_SYSTEM = "You are a careful mathematical reasoning assistant."
DIRECT_SUFFIX = "\n\nReturn only the final answer. Put it in \\boxed{...}. Do not include reasoning."
SOLUTION_SUFFIX = "\n\nSolve the problem concisely and put the final answer in \\boxed{...}."


def load_jsonl(path, max_samples=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def build_chat(tokenizer, user_prompt, enable_thinking=False, answer_mode="direct"):
    suffix = DIRECT_SUFFIX if answer_mode == "direct" else SOLUTION_SUFFIX
    messages = [
        {"role": "system", "content": BASE_SYSTEM},
        {"role": "user", "content": str(user_prompt) + suffix},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def generate_one(model, tokenizer, prompt, max_new_tokens, temperature, top_p):
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]
    with torch.inference_mode():
        gen = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(gen[0][prompt_len:], skip_special_tokens=True).strip()


def summarize(rows):
    base_total = len(rows)
    base_pass = sum(1 for x in rows if x.get("base_reward") == 1)
    correction_rows = [x for x in rows if x.get("correction_answer") is not None]
    corr_total = len(correction_rows)
    corr_pass = sum(1 for x in correction_rows if x.get("correction_reward") == 1)
    rescued = sum(1 for x in correction_rows if x.get("base_reward") == 0 and x.get("correction_reward") == 1)
    harmed = sum(1 for x in correction_rows if x.get("base_reward") == 1 and x.get("correction_reward") == 0)
    return {
        "total_samples": base_total,
        "base_pass": base_pass,
        "base_fail": base_total - base_pass,
        "base_accuracy": base_pass / base_total if base_total else 0.0,
        "correction_attempted": corr_total,
        "correction_pass": corr_pass,
        "correction_fail": corr_total - corr_pass,
        "correction_accuracy_on_attempted": corr_pass / corr_total if corr_total else 0.0,
        "rescued_failures": rescued,
        "harmed_successes": harmed,
    }


def save_state(args, out_rows):
    write_jsonl(args.output, out_rows)
    replay_rows = []
    for row in out_rows:
        if row.get("correction_prompt") is None:
            continue
        replay_rows.append({
            "id": row["id"],
            "prompt": row["correction_prompt"],
            "reference_answer": row["reference_answer"],
            "candidate_solution": row["base_answer"],
            "candidate_reward": row["base_reward"],
            "correction_answer": row["correction_answer"],
            "correction_reward": row["correction_reward"],
            "source": row["source"],
        })
    write_jsonl(args.replay_output, replay_rows)
    report = summarize(out_rows)
    report.update({
        "model": args.model,
        "input": args.input,
        "output": args.output,
        "replay_output": args.replay_output,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "correction_policy": args.correction_policy,
        "enable_thinking": args.enable_thinking,
        "answer_mode": args.answer_mode,
        "examples": [
            {
                "id": x["id"],
                "source": x["source"],
                "base_predicted_answer": x["base_predicted_answer"],
                "base_reward": x["base_reward"],
                "correction_predicted_answer": x["correction_predicted_answer"],
                "correction_reward": x["correction_reward"],
            }
            for x in out_rows[:10]
        ],
    })
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models")
    ap.add_argument("--input", default="/root/autodl-tmp/learning_from_failure_exp/data/phase_a_math_verify.jsonl")
    ap.add_argument("--output", default="/root/autodl-tmp/learning_from_failure_exp/data/phase_b_rollouts.jsonl")
    ap.add_argument("--replay-output", default="/root/autodl-tmp/learning_from_failure_exp/data/phase_b_correction_replay.jsonl")
    ap.add_argument("--report", default="/root/autodl-tmp/learning_from_failure_exp/reports/phase_b_stats.json")
    ap.add_argument("--max-samples", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--correction-policy", choices=["failures", "all"], default="failures")
    ap.add_argument("--answer-mode", choices=["direct", "solution"], default="direct")
    ap.add_argument("--enable-thinking", action="store_true")
    args = ap.parse_args()

    rows = load_jsonl(args.input, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()

    out_rows = []
    for idx, item in enumerate(rows, 1):
        base_prompt = build_chat(tokenizer, item["prompt"], enable_thinking=args.enable_thinking, answer_mode=args.answer_mode)
        base_answer = generate_one(model, tokenizer, base_prompt, args.max_new_tokens, args.temperature, args.top_p)
        base_check = verify_math_response(base_answer, item["reference_answer"])
        row = {
            "id": item["id"],
            "source": item["source"],
            "prompt": item["prompt"],
            "reference_answer": item["reference_answer"],
            "distilled_reward": item.get("distilled_reward"),
            "base_answer": base_answer,
            "base_predicted_answer": base_check["predicted_answer"],
            "base_reward": base_check["reward"],
            "correction_prompt": None,
            "correction_answer": None,
            "correction_predicted_answer": None,
            "correction_reward": None,
        }
        if args.correction_policy == "all" or row["base_reward"] == 0:
            correction_prompt_raw = CORRECTION_TEMPLATE.format(
                original_prompt=row["prompt"],
                candidate_solution=row["base_answer"],
            )
            row["correction_prompt"] = correction_prompt_raw
            correction_prompt = build_chat(
                tokenizer,
                correction_prompt_raw,
                enable_thinking=args.enable_thinking,
                answer_mode=args.answer_mode,
            )
            correction_answer = generate_one(model, tokenizer, correction_prompt, args.max_new_tokens, args.temperature, args.top_p)
            corr_check = verify_math_response(correction_answer, item["reference_answer"])
            row["correction_answer"] = correction_answer
            row["correction_predicted_answer"] = corr_check["predicted_answer"]
            row["correction_reward"] = corr_check["reward"]
        out_rows.append(row)
        report = save_state(args, out_rows)
        print(
            "[{}/{}] id={} base={} correction={} base_acc={:.3f}".format(
                idx,
                len(rows),
                row["id"],
                row["base_reward"],
                row["correction_reward"],
                report["base_accuracy"],
            ),
            flush=True,
        )

    print(json.dumps(save_state(args, out_rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
