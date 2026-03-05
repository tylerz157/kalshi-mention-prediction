#!/usr/bin/env python3
"""
predict.py -- Interactive inference with the fine-tuned model.

Usage:
  python predict.py                          # interactive mode
  python predict.py --sample 5               # show 5 random val samples with predictions vs actual
  python predict.py --prompt "We are going to build a wall"  # one-shot prediction
"""

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_DIR = Path("model")
VAL_FILE = Path("train/data/openai_val.jsonl")

def load_model():
    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB)")

        if vram_gb >= 20:
            # Full bf16 for large GPUs (A100, etc)
            model = AutoModelForCausalLM.from_pretrained(
                str(MODEL_DIR),
                dtype=torch.bfloat16,
                device_map="auto",
            )
        elif vram_gb >= 12:
            # 8-bit for mid-range GPUs (3090, 4070 Ti, etc)
            print(f"  Using 8-bit quantization")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(MODEL_DIR),
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            # Small GPU (3050, etc) -- CPU is faster than GPU offload
            print(f"  GPU too small for 8B model, using CPU (float16)")
            device = "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                str(MODEL_DIR),
                dtype=torch.float16,
                device_map="cpu",
            )
    else:
        print("  No GPU detected, using CPU")
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            dtype=torch.float16,
            device_map="cpu",
        )

    model.eval()
    print(f"  Model loaded successfully")
    return model, tokenizer, device


def generate(model, tokenizer, device, messages, max_new_tokens=200):
    import sys
    import time

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Stream tokens one at a time so user sees progress
    from transformers import TextStreamer
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print(f"(generating ~{max_new_tokens} tokens, input={input_len} tokens...)")
    start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            streamer=streamer,
        )

    elapsed = time.time() - start
    new_tokens = outputs[0][input_len:]
    tps = len(new_tokens) / elapsed if elapsed > 0 else 0
    print(f"\n({len(new_tokens)} tokens in {elapsed:.1f}s = {tps:.1f} tok/s)")

    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_sample(model, tokenizer, device, n=5):
    """Run model on random validation samples and compare to actual."""
    if not VAL_FILE.exists():
        print(f"Val file not found: {VAL_FILE}")
        return

    lines = VAL_FILE.read_text(encoding="utf-8").splitlines()
    samples = [json.loads(l) for l in lines if l.strip()]
    picks = random.sample(samples, min(n, len(samples)))

    for i, sample in enumerate(picks, 1):
        msgs = sample["messages"]
        # Extract the user prompt and actual assistant response
        user_msg = next(m["content"] for m in msgs if m["role"] == "user")
        actual = next(m["content"] for m in msgs if m["role"] == "assistant")

        # Build prompt (system + user only)
        prompt_msgs = [m for m in msgs if m["role"] != "assistant"]

        print(f"\n{'='*60}")
        print(f"SAMPLE {i}/{len(picks)}")
        print(f"{'='*60}")

        # Show full context: event, words, news, and last 300 chars of speech
        sections = user_msg.split("\n\n")
        for section in sections:
            if section.startswith("Speech so far:"):
                speech_text = section[len("Speech so far:\n"):]
                # Show last 300 chars of speech context
                if len(speech_text) > 300:
                    print(f"  Speech so far: ...{speech_text[-300:]}")
                else:
                    print(f"  Speech so far: {speech_text}")
            else:
                # Show event, words, news in full
                for line in section.split("\n"):
                    print(f"  {line}")
            print()

        print(f"--- PREDICTED ---")
        predicted = generate(model, tokenizer, device, prompt_msgs)
        print(predicted)

        print(f"\n--- ACTUAL ---")
        print(actual)

        print()


def run_interactive(model, tokenizer, device):
    """Interactive mode: type speech text, get predictions."""
    print("\nINTERACTIVE MODE")
    print("Type speech text so far, then press Enter to see what Trump says next.")
    print("Type 'quit' to exit.\n")

    system_msg = {
        "role": "system",
        "content": "You are predicting what Donald Trump will say next in a live speech. Given recent news and the verbatim speech so far, predict what he says next."
    }

    while True:
        try:
            speech = input("Speech so far> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if speech.lower() in ("quit", "exit", "q"):
            break
        if not speech:
            continue

        user_content = f"Words spoken so far: {len(speech.split())}\n\nSpeech so far:\n{speech}"
        messages = [system_msg, {"role": "user", "content": user_content}]

        print("\nPrediction:")
        result = generate(model, tokenizer, device, messages)
        print(result)
        print()


def run_prompt(model, tokenizer, device, prompt_text):
    """One-shot: predict from a given speech excerpt."""
    system_msg = {
        "role": "system",
        "content": "You are predicting what Donald Trump will say next in a live speech. Given recent news and the verbatim speech so far, predict what he says next."
    }
    user_content = f"Words spoken so far: {len(prompt_text.split())}\n\nSpeech so far:\n{prompt_text}"
    messages = [system_msg, {"role": "user", "content": user_content}]

    result = generate(model, tokenizer, device, messages)
    print(result)


def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned model")
    parser.add_argument("--sample", type=int, help="Show N random val samples with predictions vs actual")
    parser.add_argument("--prompt", type=str, help="One-shot: predict from this speech text")
    args = parser.parse_args()

    model, tokenizer, device = load_model()

    if args.sample:
        run_sample(model, tokenizer, device, n=args.sample)
    elif args.prompt:
        run_prompt(model, tokenizer, device, args.prompt)
    else:
        run_interactive(model, tokenizer, device)


if __name__ == "__main__":
    main()
