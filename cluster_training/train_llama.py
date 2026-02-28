#!/usr/bin/env python3
"""
train_llama.py -- Full fine-tune Llama 3.1 8B Instruct on Trump speech data.

Launch with torchrun on 8 GPUs:
  torchrun --nproc_per_node=8 train_llama.py

Or with accelerate:
  accelerate launch --config_file accelerate_config.yaml train_llama.py

Requirements:
  pip install transformers datasets trl accelerate deepspeed flash-attn

Set HF_TOKEN env var if Llama 3.1 requires authentication:
  export HF_TOKEN=hf_...
"""

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID      = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TRAIN_FILE    = Path("data/openai_train.jsonl")
VAL_FILE      = Path("data/openai_val.jsonl")
OUTPUT_DIR    = Path("checkpoints/llama-trump")

MAX_SEQ_LEN   = 4096   # covers ~95% of samples; longer ones are truncated
NUM_EPOCHS    = 10
LR            = 2e-5
PER_GPU_BATCH = 2      # 2 × 8 GPUs × grad_accum 4 = 64 effective batch
GRAD_ACCUM    = 4
WARMUP_STEPS  = 100
LOG_STEPS     = 25

# ── Load data ─────────────────────────────────────────────────────────────────

def load_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def apply_template(tokenizer, sample):
    """Convert messages list to a single formatted string using the model's chat template."""
    return tokenizer.apply_chat_template(
        sample['messages'],
        tokenize=False,
        add_generation_prompt=False,
    )


def build_dataset(path: Path, tokenizer):
    raw = load_jsonl(path)
    texts = [apply_template(tokenizer, s) for s in raw]
    return Dataset.from_dict({"text": texts})


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Build datasets
    print("Loading datasets...")
    train_dataset = build_dataset(TRAIN_FILE, tokenizer)
    val_dataset   = build_dataset(VAL_FILE,   tokenizer)
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")

    # Load model
    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
        use_cache=False,          # required for gradient checkpointing
    )
    model.gradient_checkpointing_enable()

    # Training arguments
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_GPU_BATCH,
        per_device_eval_batch_size=PER_GPU_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        bf16=True,
        tf32=True,
        logging_steps=LOG_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        deepspeed="deepspeed_config.json",
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Only train on assistant responses (not the prompt tokens)
    response_template = "<|start_header_id|>assistant<|end_header_id|>"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    print(f"Model saved to {OUTPUT_DIR / 'final'}")


if __name__ == "__main__":
    main()
