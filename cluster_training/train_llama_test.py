#!/usr/bin/env python3
"""
train_llama_test.py -- Smoke test version of train_llama.py

Tests the full training pipeline with a tiny model and 10 fake samples.
No GPU required, no HF token required, runs in ~30 seconds on CPU.

  python train_llama_test.py
"""

import tempfile
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)
from trl import SFTTrainer, SFTConfig

# ── Use a tiny public model (no auth needed, ~500MB) ─────────────────────────

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

# ── Generate fake data ───────────────────────────────────────────────────────

def make_fake_samples(n=10):
    samples = []
    for i in range(n):
        samples.append({
            "messages": [
                {"role": "system", "content": "You are predicting what Donald Trump will say next in a live speech."},
                {"role": "user", "content": f"Event: Test Speech {i}\n\nSpeech so far:\nWe are going to make this country great again. Sample {i}."},
                {"role": "assistant", "content": f"And we will do it better than anyone has ever done it before. Believe me. Sample {i}."},
            ]
        })
    return samples


def apply_template(tokenizer, sample):
    return tokenizer.apply_chat_template(
        sample['messages'],
        tokenize=False,
        add_generation_prompt=False,
    )


def build_dataset(samples, tokenizer):
    texts = [apply_template(tokenizer, s) for s in samples]
    return Dataset.from_dict({"text": texts})


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("SMOKE TEST - testing training pipeline")
    print("=" * 50)

    # Tokenizer
    print(f"\n[1/5] Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Fake data
    print("[2/5] Creating fake dataset (10 train, 4 val)...")
    train_ds = build_dataset(make_fake_samples(10), tokenizer)
    val_ds = build_dataset(make_fake_samples(4), tokenizer)
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")

    # Model
    print(f"[3/5] Loading model: {MODEL_ID}")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        use_cache=False,
    )
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {param_count:.0f}M parameters, dtype={dtype}")

    if torch.cuda.is_available():
        print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
        model.gradient_checkpointing_enable()
    else:
        print("  Running on CPU (no GPU needed for smoke test)")

    # Training args -- minimal, no deepspeed, works on CPU or single GPU
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"[4/5] Setting up trainer...")
        args = SFTConfig(
            output_dir=tmpdir,
            num_train_epochs=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=2e-5,
            lr_scheduler_type="cosine",
            warmup_steps=2,
            weight_decay=0.01,
            label_smoothing_factor=0.1,
            bf16=use_bf16,
            fp16=False,
            logging_steps=1,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            gradient_checkpointing=torch.cuda.is_available(),
            remove_unused_columns=False,
            use_cpu=not torch.cuda.is_available(),
            # SFT-specific
            max_length=512,
            dataset_text_field="text",
            completion_only_loss=True,
        )

        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        print("[5/5] Training (2 epochs on 10 samples)...")
        trainer.train()

        # Save test
        save_path = Path(tmpdir) / "test_final"
        trainer.save_model(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        print(f"  Model saved to temp dir: OK")

    print()
    print("=" * 50)
    print("SMOKE TEST PASSED")
    print("=" * 50)
    print()
    print("Environment is working. For the real training run:")
    print("  torchrun --nproc_per_node=8 train_llama.py")


if __name__ == "__main__":
    main()
