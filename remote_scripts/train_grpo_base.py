import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

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
                    "source": row.get("source"),
                }
            )
            if limit and len(rows) >= limit:
                break
    return Dataset.from_list(rows)


def load_eval_rows(path, limit=None, offset=0):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset or not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                {
                    "idx": row.get("idx", i),
                    "id": row.get("id", str(i)),
                    "prompt": row["prompt"],
                    "reference_answer": row["reference_answer"],
                }
            )
            if limit and len(rows) >= limit:
                break
    return rows


def math_reward(prompts, completions, reference_answer, **kwargs):
    rewards = []
    for completion, ref in zip(completions, reference_answer):
        if isinstance(completion, list):
            text = completion[-1].get("content", "") if completion else ""
        else:
            text = str(completion)
        rewards.append(float(verify_math_response(text, ref)["reward"]))
    return rewards


class EvalOnSaveCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        eval_rows,
        output_dir,
        max_prompt_length=1024,
        max_new_tokens=768,
        best_dir_name="best_eval_checkpoint",
    ):
        self.tokenizer = tokenizer
        self.eval_rows = eval_rows
        self.output_dir = Path(output_dir)
        self.max_prompt_length = max_prompt_length
        self.max_new_tokens = max_new_tokens
        self.best_dir = self.output_dir / best_dir_name
        self.history_path = self.output_dir / "eval_history.jsonl"
        self.best_metrics_path = self.best_dir / "eval_metrics.json"
        self.best_accuracy = -1.0
        if self.best_metrics_path.exists():
            try:
                self.best_accuracy = json.loads(self.best_metrics_path.read_text(encoding="utf-8")).get("accuracy", -1.0)
            except Exception:
                self.best_accuracy = -1.0

    def _chat_text(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": str(prompt).strip()}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _model_for_generate(self, model):
        if hasattr(model, "generate"):
            return model
        if hasattr(model, "module") and hasattr(model.module, "generate"):
            return model.module
        raise AttributeError("Model object does not expose generate().")

    def on_save(self, args, state, control, **kwargs):
        if not self.eval_rows:
            return control
        model = kwargs.get("model")
        if model is None:
            return control

        gen_model = self._model_for_generate(model)
        device = next(gen_model.parameters()).device
        was_training = gen_model.training
        gen_model.eval()
        correct = 0
        total_tokens = 0
        clipped = 0
        predictions = []

        try:
            for row in self.eval_rows:
                text = self._chat_text(row["prompt"])
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_prompt_length,
                ).to(device)
                with torch.inference_mode():
                    output_ids = gen_model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                input_len = int(inputs["input_ids"].shape[-1])
                new_ids = output_ids[0][input_len:]
                response = self.tokenizer.decode(new_ids, skip_special_tokens=False)
                new_token_count = int(new_ids.shape[-1])
                is_clipped = new_token_count >= self.max_new_tokens and (
                    self.tokenizer.eos_token_id is None or int(new_ids[-1]) != self.tokenizer.eos_token_id
                )
                verdict = verify_math_response(response, row["reference_answer"])
                ok = int(verdict["reward"])
                correct += ok
                total_tokens += new_token_count
                clipped += int(is_clipped)
                predictions.append(
                    {
                        "idx": row["idx"],
                        "id": row["id"],
                        "ok": ok,
                        "predicted_answer": verdict["predicted_answer"],
                        "reference_answer": row["reference_answer"],
                        "new_tokens": new_token_count,
                        "clipped": bool(is_clipped),
                    }
                )
        finally:
            if was_training:
                gen_model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        total = len(self.eval_rows)
        accuracy = correct / total if total else 0.0
        metrics = {
            "step": int(state.global_step),
            "checkpoint": str(self.output_dir / f"checkpoint-{state.global_step}"),
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "avg_new_tokens": total_tokens / total if total else 0.0,
            "clipped": clipped,
            "clipped_ratio": clipped / total if total else 0.0,
            "max_prompt_length": self.max_prompt_length,
            "max_new_tokens": self.max_new_tokens,
            "is_best": accuracy > self.best_accuracy,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        pred_path = self.output_dir / f"eval_predictions_step{state.global_step}.jsonl"
        with pred_path.open("w", encoding="utf-8") as f:
            for rec in predictions:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(json.dumps({"eval_on_save": metrics}, ensure_ascii=False), flush=True)
        if metrics["is_best"]:
            if self.best_dir.exists():
                shutil.rmtree(self.best_dir)
            self.best_dir.mkdir(parents=True, exist_ok=True)
            gen_model.save_pretrained(self.best_dir, safe_serialization=True)
            self.tokenizer.save_pretrained(self.best_dir)
            (self.best_dir / "eval_metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.best_accuracy = accuracy
        return control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", default="data/gsm8k_grpo/train.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/qwen25_05b_grpo_base_smoke")
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--report-to-tensorboard", action="store_true")
    parser.add_argument("--skip-final-save", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--eval-on-save", action="store_true")
    parser.add_argument("--eval-data", default="data/gsm8k_grpo/test.jsonl")
    parser.add_argument("--eval-limit", type=int, default=64)
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--eval-max-prompt-length", type=int, default=1024)
    parser.add_argument("--eval-max-new-tokens", type=int, default=768)
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
        reward_funcs=math_reward,
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
            "base_model": args.model,
            "data": args.data,
            "dataset_count": len(dataset),
            "output_dir": args.output_dir,
            "max_steps": args.max_steps,
            "num_generations": args.num_generations,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_completion_length,
            "lr": args.lr,
            "beta": args.beta,
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
    Path(args.output_dir, "grpo_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
