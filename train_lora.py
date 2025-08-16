#!/usr/bin/env python
# train_lora.py
import argparse, json
from dataclasses import dataclass
from typing import Dict, List
import torch
from datasets import load_dataset

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, set_seed
)


from peft import LoraConfig, get_peft_model


def build_prompt(tokenizer, prompt: str, response: str) -> str:
    # Use chat template if available (Phi-3 supports it)
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # Fallback simple format
    return f"User: {prompt}\nAssistant: {response}{tokenizer.eos_token or ''}"

def format_example(tokenizer, ex):
    text = build_prompt(tokenizer, ex["prompt"], ex["response"])
    return {"text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", default="microsoft/phi-3-mini-4k-instruct")
    ap.add_argument("--train_file", default="data/train.jsonl")
    ap.add_argument("--validation_file", default="data/validation.jsonl")
    ap.add_argument("--output_dir", default="outputs/adapters/phi3-cornell-lora")
    ap.add_argument("--per_device_train_batch_size", type=int, default=8)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--cutoff_len", type=int, default=4096, help="truncate/pad to this many tokens")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch_dtype, device_map="auto"
    )

    # LoRA config (keep as-is, or restrict to attention/MLP modules if desired)
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear",
    )
    model = get_peft_model(base_model, lora_cfg)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "config"):
            model.config.use_cache = False

    # --- Data ---
    ds = load_dataset("json", data_files={"train": args.train_file, "validation": args.validation_file})
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in ("prompt","response")])

    def _fmt(ex):
        return {"text": build_prompt(tokenizer, ex["prompt"], ex["response"])}
    ds = ds.map(_fmt, remove_columns=["prompt","response"], desc="formatting")

    def _tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=min(args.cutoff_len, tokenizer.model_max_length),
        )
    ds = ds.map(_tok, batched=True, remove_columns=["text"], desc="tokenizing")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",      # <- enable eval
        eval_steps=args.eval_steps,
        save_strategy="steps",
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        report_to="none",
        save_total_limit=2,
        group_by_length=True,             # <- bucketing speeds up training
        remove_unused_columns=False,      # <- keep tokenized cols
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
    )

    # Let Trainer auto-resume if a last checkpoint exists in output_dir
    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()