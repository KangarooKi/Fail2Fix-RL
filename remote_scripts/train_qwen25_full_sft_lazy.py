import argparse
import array
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


class IndexedJsonlSFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=8192, index_path=None):
        self.path = str(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.index_path = index_path or self.path + ".u64idx"
        self.offsets = self._load_or_build_index()
        self._fh = None

    def _load_or_build_index(self):
        idx = Path(self.index_path)
        src = Path(self.path)
        offsets = array.array("Q")
        if idx.exists() and idx.stat().st_mtime >= src.stat().st_mtime:
            with idx.open("rb") as f:
                offsets.fromfile(f, idx.stat().st_size // offsets.itemsize)
            return offsets

        pos = 0
        with src.open("rb") as f:
            for line in f:
                if line.strip():
                    offsets.append(pos)
                pos += len(line)
        with idx.open("wb") as f:
            offsets.tofile(f)
        return offsets

    def __len__(self):
        return len(self.offsets)

    def _file(self):
        if self._fh is None:
            self._fh = open(self.path, "r", encoding="utf-8")
        return self._fh

    def _row(self, index):
        fh = self._file()
        fh.seek(self.offsets[index])
        return json.loads(fh.readline())

    def __getitem__(self, index):
        for shift in range(8):
            row = self._row((index + shift) % len(self.offsets))
            item = self.encode(row)
            if item is not None:
                return item
        raise RuntimeError("Could not encode a usable sample after several attempts.")

    def encode(self, row):
        prompt = (row.get("prompt") or "").strip()
        response = (row.get("response") or row.get("distilled_answer") or "").strip()
        if not prompt or not response:
            return None
        user = {"role": "user", "content": prompt}
        assistant = {"role": "assistant", "content": response}
        prompt_text = self.tokenizer.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
        full_text = self.tokenizer.apply_chat_template([user, assistant], tokenize=False, add_generation_prompt=False)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            add_special_tokens=False,
        )
        enc.pop("token_type_ids", None)
        input_ids = enc["input_ids"]
        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        if all(x == -100 for x in labels):
            return None
        return {"input_ids": input_ids, "attention_mask": enc["attention_mask"], "labels": labels}


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [pad_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models")
    ap.add_argument("--data", default="/root/autodl-tmp/learning_from_failure_exp/data/splits/sft_train.jsonl")
    ap.add_argument("--eval-data", default="/root/autodl-tmp/learning_from_failure_exp/data/qwen25_05b_eval_100.jsonl")
    ap.add_argument("--output-dir", default="/root/autodl-tmp/learning_from_failure_exp/checkpoints/qwen25_05b_full_sft_full_m8192")
    ap.add_argument("--logging-dir", default="/root/autodl-tmp/learning_from_failure_exp/tensorboard/qwen25_05b_full_sft_full_m8192")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--eval-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--save-total-limit", type=int, default=1)
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--resume-from-checkpoint", default=None)
    args = ap.parse_args()

    os.environ.setdefault("WANDB_DISABLED", "true")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
    ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    dataset = IndexedJsonlSFTDataset(args.data, tokenizer, max_length=args.max_length)
    eval_dataset = IndexedJsonlSFTDataset(args.eval_data, tokenizer, max_length=args.max_length) if args.eval_data else None
    collator = Collator(tokenizer)
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="adamw_torch_fused",
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metrics = dict(result.metrics)
    metrics.update({
        "base_model": args.model,
        "data": args.data,
        "eval_data": args.eval_data,
        "output_dir": args.output_dir,
        "logging_dir": args.logging_dir,
        "dataset_count": len(dataset),
        "eval_dataset_count": len(eval_dataset) if eval_dataset is not None else 0,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "eval_steps": args.eval_steps,
        "metric_for_best_model": "eval_loss",
        "trainable": "full",
    })
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, "sft_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
