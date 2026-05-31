import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from verifier_math import verify_math_response

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def read_jsonl(path, limit=None, offset=0):
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
                    "question": row.get("question", ""),
                    "reference_answer": row["reference_answer"],
                }
            )
            if limit and len(rows) >= limit:
                break
    return rows


def chat_text(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": str(prompt).strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )


def correction_prompt(original_prompt, candidate_solution):
    return (
        "You are solving a grade-school math problem. A previous candidate solution is provided and may be wrong.\n"
        "Check the candidate solution, fix any mistake, and give a concise corrected solution.\n\n"
        "Use exactly this output format:\n"
        "<error>\n"
        "One or two sentences explaining the candidate's main mistake. If it is already correct, say it is correct.\n"
        "</error>\n"
        "<corrected_solution>\n"
        "Concise corrected reasoning.\n"
        "</corrected_solution>\n"
        "<final_answer>the final number only</final_answer>\n"
        "Answer: the final number only\n\n"
        "The final line must begin with `Answer:` and contain only the final numeric answer after it. "
        "Do not use boxed answers or the GSM8K #### marker.\n\n"
        "<problem>\n"
        f"{original_prompt.strip()}\n"
        "</problem>\n\n"
        "<candidate_solution>\n"
        f"{str(candidate_solution).strip()}\n\n"
        "</candidate_solution>\n"
    )


def tokenize_prompt(tokenizer, prompt, max_prompt_length):
    text = chat_text(tokenizer, prompt)
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=False,
    )
    return encoded["input_ids"], text


def trim_completion_ids(ids, pad_token_id):
    ids = list(map(int, ids))
    if pad_token_id is not None:
        while ids and ids[-1] == pad_token_id:
            ids.pop()
    return ids


def set_generation_mode(model, enabled):
    if not hasattr(model, "config"):
        return None
    old_use_cache = getattr(model.config, "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = bool(enabled)
    return old_use_cache


def restore_use_cache(model, old_use_cache):
    if old_use_cache is not None and hasattr(model, "config"):
        model.config.use_cache = old_use_cache


def generate_groups(model, tokenizer, prompts, group_size, args, device):
    groups = []
    was_training = model.training
    old_use_cache = set_generation_mode(model, True)
    model.eval()
    try:
        for start in range(0, len(prompts), args.generation_batch_size):
            batch_prompts = prompts[start : start + args.generation_batch_size]
            texts = [chat_text(tokenizer, p) for p in batch_prompts]
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_length,
                add_special_tokens=False,
            ).to(device)
            input_len = int(encoded["input_ids"].shape[-1])
            with torch.inference_mode():
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=group_size,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_ids_by_row = []
            for row_idx in range(len(batch_prompts)):
                mask = encoded["attention_mask"][row_idx].bool()
                prompt_ids_by_row.append(encoded["input_ids"][row_idx][mask].detach().cpu().tolist())

            for row_idx in range(len(batch_prompts)):
                samples = []
                prompt_ids = prompt_ids_by_row[row_idx]
                for sample_idx in range(group_size):
                    out_idx = row_idx * group_size + sample_idx
                    new_ids = trim_completion_ids(output_ids[out_idx][input_len:].detach().cpu().tolist(), tokenizer.pad_token_id)
                    text = tokenizer.decode(new_ids, skip_special_tokens=True)
                    clipped = len(new_ids) >= args.max_new_tokens and (
                        tokenizer.eos_token_id is None or (new_ids and int(new_ids[-1]) != tokenizer.eos_token_id)
                    )
                    samples.append(
                        {
                            "prompt_ids": prompt_ids,
                            "completion_ids": new_ids,
                            "text": text,
                            "new_tokens": len(new_ids),
                            "clipped": bool(clipped),
                        }
                    )
                groups.append(samples)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        restore_use_cache(model, old_use_cache)
        if was_training:
            model.train()
    return groups


def score_base_groups(rows, groups):
    scored_groups = []
    for row, samples in zip(rows, groups):
        scored = []
        for sample_idx, sample in enumerate(samples):
            verdict = verify_math_response(sample["text"], row["reference_answer"])
            rec = dict(sample)
            rec.update(
                {
                    "row": row,
                    "sample_idx": sample_idx,
                    "reward": float(verdict["reward"]),
                    "predicted_answer": verdict["predicted_answer"],
                }
            )
            scored.append(rec)
        pass_rate = sum(x["reward"] for x in scored) / max(1, len(scored))
        scored_groups.append({"row": row, "samples": scored, "pass_rate": pass_rate})
    return scored_groups


def select_correction_anchors(scored_groups, rho, args, rng):
    target_n = max(1, int(math.floor(args.replay_fraction * len(scored_groups))))
    medium_groups = [g for g in scored_groups if args.delta_low <= g["pass_rate"] <= args.delta_high]
    fallback_groups = [g for g in scored_groups if not (args.delta_low <= g["pass_rate"] <= args.delta_high)]

    def flatten(groups):
        candidates = []
        for group in groups:
            for sample in group["samples"]:
                candidates.append(
                    {
                        "row": group["row"],
                        "candidate_solution": sample["text"],
                        "anchor_reward": int(sample["reward"]),
                        "anchor_new_tokens": sample["new_tokens"],
                        "anchor_clipped": sample["clipped"],
                        "pass_rate": group["pass_rate"],
                    }
                )
        rng.shuffle(candidates)
        return candidates

    ordered = flatten(medium_groups) + flatten(fallback_groups)
    positives = [rec for rec in ordered if rec["anchor_reward"] == 1]
    negatives = [rec for rec in ordered if rec["anchor_reward"] == 0]

    pos_target = min(len(positives), int(math.floor(rho * target_n)))
    neg_target = min(len(negatives), target_n - pos_target)
    anchors = positives[:pos_target] + negatives[:neg_target]
    if len(anchors) < target_n:
        anchors.extend(positives[pos_target : pos_target + (target_n - len(anchors))])
    rng.shuffle(anchors)
    return anchors, {
        "target": target_n,
        "selected": len(anchors),
        "selected_correct": sum(1 for x in anchors if x["anchor_reward"] == 1),
        "selected_failed": sum(1 for x in anchors if x["anchor_reward"] == 0),
        "medium_prompts": len(medium_groups),
        "fallback_prompts": len(fallback_groups),
        "medium_candidates": sum(len(g["samples"]) for g in medium_groups),
        "fallback_candidates": sum(len(g["samples"]) for g in fallback_groups),
    }


def make_examples_from_base(scored_groups):
    examples = []
    for group_idx, group in enumerate(scored_groups):
        row = group["row"]
        for sample in group["samples"]:
            if not sample["completion_ids"]:
                continue
            examples.append(
                {
                    "stream": "base",
                    "group_id": f"base::{row['id']}::{group_idx}",
                    "prompt_ids": sample["prompt_ids"],
                    "completion_ids": sample["completion_ids"],
                    "response": sample["text"],
                    "reward": float(sample["reward"]),
                    "anchor_reward": -1,
                    "row_id": row["id"],
                    "new_tokens": sample["new_tokens"],
                    "clipped": sample["clipped"],
                }
            )
    return examples


def make_correction_examples(tokenizer, anchors, groups, args):
    examples = []
    for anchor_idx, (anchor, samples) in enumerate(zip(anchors, groups)):
        row = anchor["row"]
        for sample_idx, sample in enumerate(samples):
            if not sample["completion_ids"]:
                continue
            verdict = verify_math_response(sample["text"], row["reference_answer"])
            base_reward = float(verdict["reward"])
            reward = base_reward
            if anchor["anchor_reward"] == 1 and base_reward < 1.0:
                reward -= float(args.risk_lambda)
            examples.append(
                {
                    "stream": "correction",
                    "group_id": f"corr::{row['id']}::{anchor_idx}",
                    "prompt_ids": sample["prompt_ids"],
                    "completion_ids": sample["completion_ids"],
                    "response": sample["text"],
                    "reward": float(reward),
                    "raw_reward": base_reward,
                    "anchor_reward": int(anchor["anchor_reward"]),
                    "row_id": row["id"],
                    "new_tokens": sample["new_tokens"],
                    "clipped": sample["clipped"],
                }
            )
    return examples


def add_group_advantages(examples):
    by_group = {}
    for idx, ex in enumerate(examples):
        by_group.setdefault(ex["group_id"], []).append(idx)
    for indices in by_group.values():
        rewards = torch.tensor([examples[i]["reward"] for i in indices], dtype=torch.float32)
        mean = rewards.mean()
        std = rewards.std(unbiased=False)
        if float(std) < 1e-8:
            adv = torch.zeros_like(rewards)
        else:
            adv = (rewards - mean) / (std + 1e-8)
        for pos, idx in enumerate(indices):
            examples[idx]["advantage"] = float(adv[pos])
    return examples


def pad_inputs(tokenizer, examples, device):
    lengths = [len(ex["prompt_ids"]) + len(ex["completion_ids"]) for ex in examples]
    max_len = max(lengths)
    input_ids = []
    attention = []
    prompt_lens = []
    completion_lens = []
    for ex, length in zip(examples, lengths):
        ids = ex["prompt_ids"] + ex["completion_ids"]
        pad = max_len - length
        input_ids.append(ids + [tokenizer.pad_token_id] * pad)
        attention.append([1] * length + [0] * pad)
        prompt_lens.append(len(ex["prompt_ids"]))
        completion_lens.append(len(ex["completion_ids"]))
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention, dtype=torch.long, device=device),
        prompt_lens,
        completion_lens,
    )


def compute_logprobs(model, tokenizer, examples, device, require_grad=False):
    input_ids, attention, prompt_lens, completion_lens = pad_inputs(tokenizer, examples, device)
    ctx = torch.enable_grad() if require_grad else torch.inference_mode()
    with ctx:
        outputs = model(input_ids=input_ids, attention_mask=attention)
        logits = outputs.logits[:, :-1, :].float()
        target_ids = input_ids[:, 1:]
        token_logps = F.log_softmax(logits, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    out = []
    for i, (prompt_len, completion_len) in enumerate(zip(prompt_lens, completion_lens)):
        start = max(0, prompt_len - 1)
        end = start + completion_len
        out.append(token_logps[i, start:end])
    return out


def attach_old_and_ref_logprobs(model, ref_model, tokenizer, examples, args, device):
    if not examples:
        return examples
    was_training = model.training
    old_use_cache = set_generation_mode(model, False)
    model.eval()
    try:
        for start in range(0, len(examples), args.forward_batch_size):
            batch = examples[start : start + args.forward_batch_size]
            old_logps = compute_logprobs(model, tokenizer, batch, device, require_grad=False)
            ref_logps = compute_logprobs(ref_model, tokenizer, batch, device, require_grad=False) if ref_model else old_logps
            for ex, old_lp, ref_lp in zip(batch, old_logps, ref_logps):
                ex["old_logps"] = old_lp.detach().cpu()
                ex["ref_logps"] = ref_lp.detach().cpu()
    finally:
        restore_use_cache(model, old_use_cache)
        if was_training:
            model.train()
    return examples


def pad_logprob_list(logps, device):
    max_len = max(int(x.numel()) for x in logps)
    values = []
    mask = []
    for lp in logps:
        lp = lp.to(device)
        pad = max_len - int(lp.numel())
        values.append(F.pad(lp, (0, pad), value=0.0))
        mask.append(F.pad(torch.ones_like(lp, dtype=torch.float32), (0, pad), value=0.0))
    return torch.stack(values, dim=0), torch.stack(mask, dim=0)


def backward_stream(model, tokenizer, examples, args, device, objective_weight):
    if not examples:
        return {"loss": 0.0, "kl": 0.0, "clip_ratio": 0.0, "tokens": 0}
    total_examples = len(examples)
    loss_total = 0.0
    kl_total = 0.0
    clip_total = 0.0
    token_total = 0.0
    for start in range(0, total_examples, args.forward_batch_size):
        batch = examples[start : start + args.forward_batch_size]
        new_logps = compute_logprobs(model, tokenizer, batch, device, require_grad=True)
        old_logps, mask = pad_logprob_list([ex["old_logps"] for ex in batch], device)
        ref_logps, _ = pad_logprob_list([ex["ref_logps"] for ex in batch], device)
        new_logps, _ = pad_logprob_list(new_logps, device)

        advantages = torch.tensor([ex["advantage"] for ex in batch], dtype=torch.float32, device=device).unsqueeze(1)
        ratio = torch.exp(new_logps - old_logps)
        clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
        pg_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages)
        kl = torch.exp(ref_logps - new_logps) - (ref_logps - new_logps) - 1.0
        per_token = pg_loss + args.beta * kl
        seq_loss = (per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        loss = objective_weight * seq_loss.sum() / max(1, total_examples)
        loss.backward()

        with torch.no_grad():
            tokens = float(mask.sum().detach().cpu())
            loss_total += float(seq_loss.sum().detach().cpu())
            kl_total += float((kl * mask).sum().detach().cpu())
            clip_total += float(((torch.abs(ratio - 1.0) > args.clip_range).float() * mask).sum().detach().cpu())
            token_total += tokens
        del new_logps, old_logps, ref_logps, mask, advantages, ratio, clipped_ratio, pg_loss, kl, per_token, seq_loss, loss
    return {
        "loss": loss_total / max(1, total_examples),
        "kl": kl_total / max(1.0, token_total),
        "clip_ratio": clip_total / max(1.0, token_total),
        "tokens": int(token_total),
    }


def evaluate(model, tokenizer, rows, args, device, step, output_dir):
    if not rows:
        return None
    was_training = model.training
    old_use_cache = set_generation_mode(model, True)
    model.eval()
    correct = 0
    total_tokens = 0
    clipped = 0
    predictions = []
    started = time.time()
    try:
        for row in rows:
            text = chat_text(tokenizer, row["prompt"])
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.eval_max_prompt_length,
                add_special_tokens=False,
            ).to(device)
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.eval_max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            input_len = int(inputs["input_ids"].shape[-1])
            new_ids = output_ids[0][input_len:].detach().cpu().tolist()
            text_out = tokenizer.decode(trim_completion_ids(new_ids, tokenizer.pad_token_id), skip_special_tokens=True)
            new_token_count = len(trim_completion_ids(new_ids, tokenizer.pad_token_id))
            is_clipped = new_token_count >= args.eval_max_new_tokens and (
                tokenizer.eos_token_id is None or (new_ids and int(new_ids[-1]) != tokenizer.eos_token_id)
            )
            verdict = verify_math_response(text_out, row["reference_answer"])
            ok = int(verdict["reward"])
            correct += ok
            total_tokens += new_token_count
            clipped += int(is_clipped)
            predictions.append(
                {
                    "id": row["id"],
                    "ok": ok,
                    "predicted_answer": verdict["predicted_answer"],
                    "reference_answer": row["reference_answer"],
                    "new_tokens": new_token_count,
                    "clipped": bool(is_clipped),
                }
            )
    finally:
        restore_use_cache(model, old_use_cache)
        if was_training:
            model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = {
        "step": int(step),
        "accuracy": correct / len(rows),
        "correct": correct,
        "total": len(rows),
        "avg_new_tokens": total_tokens / len(rows),
        "clipped": clipped,
        "clipped_ratio": clipped / len(rows),
        "elapsed_sec": time.time() - started,
        "max_prompt_length": args.eval_max_prompt_length,
        "max_new_tokens": args.eval_max_new_tokens,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "eval_history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
    with (output_dir / f"eval_predictions_step{step}.jsonl").open("w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return metrics


def update_rho(rho, prev_retention, retention, below_count, args):
    if retention is None:
        return rho, prev_retention, below_count
    if retention < args.retention_target:
        below_count += 1
    else:
        below_count = 0
    f1 = args.retention_target - retention
    f2 = max(0.0, (prev_retention if prev_retention is not None else retention) - retention)
    f3 = min(float(below_count), 3.0)
    multiplier = 1.0 + args.rho_w1 * f1 + args.rho_w2 * f2 + args.rho_w3 * f3
    rho = max(args.rho_min, min(args.rho_max, rho * multiplier))
    return rho, retention, below_count


def save_eval_checkpoint(model, tokenizer, output_dir, metrics, name):
    ckpt_dir = output_dir / name
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    old_use_cache = set_generation_mode(model, True)
    model.save_pretrained(ckpt_dir, safe_serialization=True)
    restore_use_cache(model, old_use_cache)
    tokenizer.save_pretrained(ckpt_dir)
    (ckpt_dir / "eval_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/models")
    parser.add_argument("--train-data", default="/root/autodl-tmp/learning_from_failure_exp/data/gsm8k_grpo/train.jsonl")
    parser.add_argument("--eval-data", default="/root/autodl-tmp/learning_from_failure_exp/data/gsm8k_grpo/test.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=64)
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--replay-fraction", type=float, default=1.0)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--forward-batch-size", type=int, default=1)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--eval-max-prompt-length", type=int, default=1024)
    parser.add_argument("--eval-max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--correction-lambda", type=float, default=1.0)
    parser.add_argument("--risk-lambda", type=float, default=1.0)
    parser.add_argument("--rho0", type=float, default=0.3)
    parser.add_argument("--rho-min", type=float, default=0.2)
    parser.add_argument("--rho-max", type=float, default=0.8)
    parser.add_argument("--retention-target", type=float, default=0.8)
    parser.add_argument("--rho-w1", type=float, default=0.8)
    parser.add_argument("--rho-w2", type=float, default=0.3)
    parser.add_argument("--rho-w3", type=float, default=0.05)
    parser.add_argument("--delta-low", type=float, default=3.0 / 8.0)
    parser.add_argument("--delta-high", type=float, default=6.0 / 8.0)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--tensorboard", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(args.train_data, args.train_limit if args.train_limit > 0 else None)
    eval_rows = read_jsonl(args.eval_data, args.eval_limit if args.eval_limit > 0 else None, args.eval_offset)
    if not train_rows:
        raise ValueError(f"No train rows loaded from {args.train_data}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    model.train()
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir=str(output_dir / "runs")) if args.tensorboard and SummaryWriter else None
    rng = random.Random(args.seed)
    rho = args.rho0
    prev_retention = None
    below_count = 0
    best_acc = -1.0
    train_log_path = output_dir / "train_history.jsonl"
    config = vars(args).copy()
    config.update({"method": "online_cipo_grpo", "device": device, "train_count": len(train_rows), "eval_count": len(eval_rows)})
    (output_dir / "cipo_online_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    for local_step in range(1, args.max_steps + 1):
        step = int(args.step_offset) + local_step
        step_started = time.time()
        batch_rows = rng.sample(train_rows, k=min(args.batch_size, len(train_rows)))

        base_groups = generate_groups(model, tokenizer, [r["prompt"] for r in batch_rows], args.group_size, args, device)
        scored_groups = score_base_groups(batch_rows, base_groups)
        base_examples = add_group_advantages(make_examples_from_base(scored_groups))

        anchors, replay_stats = select_correction_anchors(scored_groups, rho, args, rng)
        correction_prompts = [correction_prompt(a["row"]["prompt"], a["candidate_solution"]) for a in anchors]
        corr_groups = generate_groups(model, tokenizer, correction_prompts, args.group_size, args, device) if correction_prompts else []
        corr_examples = add_group_advantages(make_correction_examples(tokenizer, anchors, corr_groups, args))

        base_examples = attach_old_and_ref_logprobs(model, ref_model, tokenizer, base_examples, args, device)
        corr_examples = attach_old_and_ref_logprobs(model, ref_model, tokenizer, corr_examples, args, device)

        model.train()
        set_generation_mode(model, False)
        optimizer.zero_grad(set_to_none=True)
        base_train = backward_stream(model, tokenizer, base_examples, args, device, objective_weight=1.0)
        corr_train = backward_stream(model, tokenizer, corr_examples, args, device, objective_weight=args.correction_lambda)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        corr_shaped_from_success = [
            ex["reward"]
            for ex in corr_examples
            if ex["stream"] == "correction" and ex["anchor_reward"] == 1 and "raw_reward" in ex
        ]
        corr_raw_from_success = [
            ex["raw_reward"]
            for ex in corr_examples
            if ex["stream"] == "correction" and ex["anchor_reward"] == 1 and "raw_reward" in ex
        ]
        retention = (
            sum(corr_raw_from_success) / len(corr_raw_from_success)
            if corr_raw_from_success
            else None
        )
        rho, prev_retention, below_count = update_rho(rho, prev_retention, retention, below_count, args)

        base_rewards = [ex["reward"] for ex in base_examples]
        corr_rewards = [ex["reward"] for ex in corr_examples]
        record = {
            "step": step,
            "local_step": local_step,
            "step_offset": int(args.step_offset),
            "rho": rho,
            "retention": retention,
            "success_anchor_shaped_reward_mean": (
                sum(corr_shaped_from_success) / len(corr_shaped_from_success) if corr_shaped_from_success else None
            ),
            "success_anchor_raw_accuracy": (
                sum(corr_raw_from_success) / len(corr_raw_from_success) if corr_raw_from_success else None
            ),
            "below_retention_count": below_count,
            "base_reward_mean": sum(base_rewards) / max(1, len(base_rewards)),
            "base_reward_std": float(torch.tensor(base_rewards).std(unbiased=False)) if base_rewards else 0.0,
            "correction_reward_mean": sum(corr_rewards) / max(1, len(corr_rewards)),
            "correction_raw_reward_mean": sum(ex.get("raw_reward", 0.0) for ex in corr_examples) / max(1, len(corr_examples)),
            "base_examples": len(base_examples),
            "correction_examples": len(corr_examples),
            "base_avg_pass_rate": sum(g["pass_rate"] for g in scored_groups) / max(1, len(scored_groups)),
            "mixed_prompt_count": sum(1 for g in scored_groups if 0.0 < g["pass_rate"] < 1.0),
            "all_wrong_prompt_count": sum(1 for g in scored_groups if g["pass_rate"] == 0.0),
            "all_correct_prompt_count": sum(1 for g in scored_groups if g["pass_rate"] == 1.0),
            "replay": replay_stats,
            "base_loss": base_train["loss"],
            "correction_loss": corr_train["loss"],
            "base_kl": base_train["kl"],
            "correction_kl": corr_train["kl"],
            "base_clip_ratio": base_train["clip_ratio"],
            "correction_clip_ratio": corr_train["clip_ratio"],
            "avg_base_new_tokens": sum(ex["new_tokens"] for ex in base_examples) / max(1, len(base_examples)),
            "avg_correction_new_tokens": sum(ex["new_tokens"] for ex in corr_examples) / max(1, len(corr_examples)),
            "base_clipped_ratio": sum(int(ex["clipped"]) for ex in base_examples) / max(1, len(base_examples)),
            "correction_clipped_ratio": sum(int(ex["clipped"]) for ex in corr_examples) / max(1, len(corr_examples)),
            "grad_norm": float(grad_norm.detach().cpu()) if hasattr(grad_norm, "detach") else float(grad_norm),
            "elapsed_sec": time.time() - step_started,
        }
        with train_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if writer:
            for key, value in record.items():
                if isinstance(value, (int, float)) and value is not None:
                    writer.add_scalar(f"train/{key}", value, step)
            writer.flush()
        if step % args.logging_steps == 0:
            print(json.dumps({"train": record}, ensure_ascii=False), flush=True)

        if args.eval_steps > 0 and local_step % args.eval_steps == 0:
            metrics = evaluate(model, tokenizer, eval_rows, args, device, step, output_dir)
            if metrics:
                metrics["is_best"] = metrics["accuracy"] > best_acc
                if writer:
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)) and value is not None:
                            writer.add_scalar(f"eval/{key}", value, step)
                    writer.flush()
                print(json.dumps({"eval": metrics}, ensure_ascii=False), flush=True)
                save_eval_checkpoint(model, tokenizer, output_dir, metrics, "last_eval_checkpoint")
                (output_dir / "last_step.txt").write_text(str(step), encoding="utf-8")
                if metrics["is_best"]:
                    best_acc = metrics["accuracy"]
                    save_eval_checkpoint(model, tokenizer, output_dir, metrics, "best_eval_checkpoint")
                    (output_dir / "best_step.txt").write_text(str(step), encoding="utf-8")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
