import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("/root/autodl-tmp/learning_from_failure_exp/src")
from verifier_math import verify_math_response


def load_rows(path, limit):
    rows = []
    for line in open(path, "r", encoding="utf-8"):
        if line.strip():
            rows.append(json.loads(line))
        if len(rows) >= limit:
            break
    return rows


def build_prompt(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt.strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )


def load_model(base_model, adapter=None):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def run(model, tokenizer, rows, max_new_tokens):
    outs = []
    for row in rows:
        text = build_prompt(tokenizer, row["prompt"])
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        enc.pop("token_type_ids", None)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(gen[0][prompt_len:], skip_special_tokens=True).strip()
        check = verify_math_response(completion, row["reference_answer"])
        outs.append({
            "id": row.get("id"),
            "reference_answer": row.get("reference_answer"),
            "completion": completion,
            "predicted_answer": check["predicted_answer"],
            "reward": check["reward"],
        })
    return outs


def summarize(items):
    return {
        "reward_sum": sum(x["reward"] for x in items),
        "count": len(items),
        "reward_rate": sum(x["reward"] for x in items) / max(1, len(items)),
        "empty_pred": sum(1 for x in items if not x["predicted_answer"].strip()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="/root/models")
    ap.add_argument("--adapter", default="/root/autodl-tmp/learning_from_failure_exp/checkpoints/qwen25_05b_lora_sft_2k")
    ap.add_argument("--data", default="/root/autodl-tmp/learning_from_failure_exp/data/qwen25_05b_eval_100.jsonl")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--output", default="/root/autodl-tmp/learning_from_failure_exp/reports/qwen25_05b_lora_sft_eval.json")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = load_rows(args.data, args.limit)

    base_model = load_model(args.base_model)
    base = run(base_model, tokenizer, rows, args.max_new_tokens)
    del base_model
    torch.cuda.empty_cache()

    sft_model = load_model(args.base_model, args.adapter)
    sft = run(sft_model, tokenizer, rows, args.max_new_tokens)

    report = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "data": args.data,
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "base": summarize(base),
        "sft": summarize(sft),
        "examples": [
            {
                "id": rows[i].get("id"),
                "reference": rows[i].get("reference_answer"),
                "base_pred": base[i]["predicted_answer"],
                "base_reward": base[i]["reward"],
                "sft_pred": sft[i]["predicted_answer"],
                "sft_reward": sft[i]["reward"],
                "sft_preview": sft[i]["completion"][:300],
            }
            for i in range(min(12, len(rows)))
        ],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
