import argparse
import json
import os
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from train_dapo_grpo_base import EvalOnSaveCallback, load_eval_rows
from verifier_math import verify_math_response


def load_rows(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                {
                    "prompt": [{"role": "user", "content": row["prompt"]}],
                    "reference_answer": row["reference_answer"],
                    "id": row.get("id"),
                    "source_id": row.get("source_id"),
                    "stream": row.get("stream", "base"),
                    "anchor_reward": int(row.get("anchor_reward", -1)),
                    "empirical_pass_rate": -1.0
                    if row.get("empirical_pass_rate") is None
                    else float(row.get("empirical_pass_rate")),
                }
            )
            if limit and len(rows) >= limit:
                break
    return Dataset.from_list(rows)


def scalar_at(values, idx, default=None):
    if values is None:
        return default
    if isinstance(values, (list, tuple)):
        return values[idx] if idx < len(values) else default
    return values


def make_cipo_reward(risk_lambda):
    def cipo_reward(prompts, completions, reference_answer, stream=None, anchor_reward=None, **kwargs):
        rewards = []
        for i, (completion, ref) in enumerate(zip(completions, reference_answer)):
            if isinstance(completion, list):
                text = completion[-1].get("content", "") if completion else ""
            else:
                text = str(completion)
            base_reward = float(verify_math_response(text, ref)["reward"])
            cur_stream = scalar_at(stream, i, "base")
            cur_anchor_reward = int(scalar_at(anchor_reward, i, -1))
            reward = base_reward
            if cur_stream == "correction" and cur_anchor_reward == 1 and base_reward < 1.0:
                reward -= float(risk_lambda)
            rewards.append(reward)
        return rewards

    cipo_reward.__name__ = "cipo_math_reward"
    return cipo_reward


def stream_counts(dataset):
    counts = {"base": 0, "correction": 0, "correction_failed_anchor": 0, "correction_correct_anchor": 0}
    for row in dataset:
        if row["stream"] == "base":
            counts["base"] += 1
        elif row["stream"] == "correction":
            counts["correction"] += 1
            if int(row["anchor_reward"]) == 1:
                counts["correction_correct_anchor"] += 1
            else:
                counts["correction_failed_anchor"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/models")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--beta", type=float, default=5e-4)
    parser.add_argument("--risk-lambda", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--report-to-tensorboard", action="store_true")
    parser.add_argument("--skip-final-save", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--eval-on-save", action="store_true")
    parser.add_argument("--eval-data", default="/root/autodl-tmp/learning_from_failure_exp/data/gsm8k_grpo/test.jsonl")
    parser.add_argument("--eval-limit", type=int, default=64)
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--eval-max-prompt-length", type=int, default=1024)
    parser.add_argument("--eval-max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    os.environ.setdefault("WANDB_DISABLED", "true")
    dataset = load_rows(args.data, args.limit if args.limit > 0 else None)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eval_rows = []
    if args.eval_on_save:
        eval_rows = load_eval_rows(args.eval_data, args.eval_limit if args.eval_limit > 0 else None, args.eval_offset)

    report_to = ["tensorboard"] if args.report_to_tensorboard else []
    config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        learning_rate=args.lr,
        beta=args.beta,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        logging_steps=args.logging_steps,
        save_strategy="no" if args.skip_final_save else "steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=True,
        report_to=report_to,
        remove_unused_columns=False,
        temperature=args.temperature,
        top_p=args.top_p,
        scale_rewards=True,
        loss_type="bnpo",
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=make_cipo_reward(args.risk_lambda),
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    if args.eval_on_save:
        trainer.add_callback(
            EvalOnSaveCallback(
                tokenizer=tokenizer,
                eval_rows=eval_rows,
                output_dir=args.output_dir,
                max_prompt_length=args.eval_max_prompt_length,
                max_new_tokens=args.eval_max_new_tokens,
            )
        )

    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if not args.skip_final_save:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    metrics = dict(result.metrics)
    metrics.update(
        {
            "method": "cipo_grpo_mixed_replay",
            "base_model": args.model,
            "data": args.data,
            "dataset_count": len(dataset),
            "stream_counts": stream_counts(dataset),
            "output_dir": args.output_dir,
            "max_steps": args.max_steps,
            "num_generations": args.num_generations,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_completion_length,
            "lr": args.lr,
            "beta": args.beta,
            "risk_lambda": args.risk_lambda,
            "trainable": "full",
            "skip_final_save": args.skip_final_save,
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "eval_on_save": args.eval_on_save,
            "eval_data": args.eval_data if args.eval_on_save else None,
            "eval_count": len(eval_rows),
            "eval_max_prompt_length": args.eval_max_prompt_length if args.eval_on_save else None,
            "eval_max_new_tokens": args.eval_max_new_tokens if args.eval_on_save else None,
        }
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, "cipo_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
