import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


class ResponseOnlySFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=2048, limit=None):
        self.items = []
        self.skipped = 0
        self.tokenizer = tokenizer
        self.max_length = max_length
        for line in open(path, "r", encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            encoded = self.encode(row)
            if encoded is None:
                self.skipped += 1
                continue
            self.items.append(encoded)
            if limit and len(self.items) >= limit:
                break

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def encode(self, row):
        user = {"role": "user", "content": row["prompt"].strip()}
        assistant = {"role": "assistant", "content": row["response"].strip()}
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
    ap.add_argument("--data", default="/root/autodl-tmp/learning_from_failure_exp/data/qwen25_05b_sft_train_2k.jsonl")
    ap.add_argument("--output-dir", default="/root/autodl-tmp/learning_from_failure_exp/checkpoints/qwen25_05b_full_sft_2k")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
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

    dataset = ResponseOnlySFTDataset(args.data, tokenizer, max_length=args.max_length, limit=args.limit)
    collator = Collator(tokenizer)
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="steps",
        save_steps=args.max_steps,
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = dict(result.metrics)
    metrics.update({
        "base_model": args.model,
        "data": args.data,
        "output_dir": args.output_dir,
        "dataset_count": len(dataset),
        "dataset_skipped": dataset.skipped,
        "max_length": args.max_length,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "trainable": "full",
    })
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, "sft_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
