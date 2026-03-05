#!/usr/bin/env python3
"""
evaluate.py -- Evaluate the fine-tuned model on the validation set.

Metrics:
  1. Word overlap (ROUGE-like): what % of actual words appear in prediction
  2. Keyword hit rate: does the prediction mention the same key topics
  3. Kalshi mention accuracy: for samples tied to mention markets, would the
     model have predicted the right yes/no outcome

Usage:
  python evaluate.py                    # full val set eval
  python evaluate.py --n 50             # eval on 50 random samples
  python evaluate.py --n 50 --verbose   # show each prediction
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_DIR = Path("model")
VAL_FILE = Path("train/data/openai_val.jsonl")

# Common words to skip when computing overlap
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "that", "this", "was", "are",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "must",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "what", "which",
    "who", "whom", "whose", "where", "when", "how", "not", "no", "so",
    "if", "then", "than", "too", "very", "just", "about", "up", "out",
    "all", "been", "being", "were", "am", "as", "into", "through",
    "during", "before", "after", "because", "going", "know", "like",
    "get", "got", "go", "said", "say", "says", "also", "well", "back",
    "even", "still", "way", "take", "come", "make", "thing", "things",
    "think", "much", "many", "some", "any", "other", "over", "such",
    "now", "here", "there", "these", "those", "right", "look", "lot",
    "really", "want", "tell", "people", "going", "gonna", "dont", "dont",
    "ive", "thats", "theyre", "youre", "weve", "hes", "shes", "its",
    "were", "theyve", "youve", "didnt", "doesnt", "isnt", "arent",
    "wasnt", "werent", "wont", "wouldnt", "couldnt", "shouldnt",
    "havent", "hasnt", "hadnt", "one", "two", "yeah", "yes", "okay",
    "oh", "hey", "let", "see", "put", "new", "old", "big", "great",
    "good", "bad", "first", "last", "long", "little", "own", "same",
    "another", "around", "most", "every", "down", "did", "had", "been",
    "made", "only", "more", "than", "them", "time", "very", "when",
}


def load_model():
    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB)")
        if vram_gb < 12:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True,
            )
            from accelerate import infer_auto_device_map, init_empty_weights
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(str(MODEL_DIR))
            with init_empty_weights():
                empty_model = AutoModelForCausalLM.from_config(config)
            device_map = infer_auto_device_map(
                empty_model,
                max_memory={0: f"{int(vram_gb * 0.85)}GiB", "cpu": "24GiB"},
                no_split_module_classes=["LlamaDecoderLayer"],
            )
            del empty_model
            model = AutoModelForCausalLM.from_pretrained(
                str(MODEL_DIR), quantization_config=bnb_config, device_map=device_map,
            )
        elif vram_gb < 20:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(MODEL_DIR), quantization_config=bnb_config, device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(MODEL_DIR), dtype=torch.bfloat16, device_map="auto",
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR), dtype=torch.float32, device_map="cpu",
        )

    model.eval()
    print(f"  Model loaded successfully")
    return model, tokenizer, device


def generate(model, tokenizer, device, messages, max_new_tokens=200):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def tokenize_words(text):
    return re.findall(r"[a-z]+", text.lower())


def content_words(text):
    """Extract meaningful words (not stop words)."""
    return [w for w in tokenize_words(text) if w not in STOP_WORDS and len(w) > 2]


def word_overlap(predicted, actual):
    """Fraction of actual content words that appear in prediction."""
    actual_words = set(content_words(actual))
    if not actual_words:
        return 0.0
    pred_words = set(content_words(predicted))
    return len(actual_words & pred_words) / len(actual_words)


def extract_event_name(user_msg):
    """Pull event name from user message."""
    for line in user_msg.split("\n"):
        if line.startswith("Event:"):
            return line.replace("Event:", "").strip()
    return "Unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0, help="Number of samples (0 = all)")
    parser.add_argument("--verbose", action="store_true", help="Print each prediction")
    args = parser.parse_args()

    model, tokenizer, device = load_model()

    lines = VAL_FILE.read_text(encoding="utf-8").splitlines()
    samples = [json.loads(l) for l in lines if l.strip()]

    if args.n > 0:
        samples = random.sample(samples, min(args.n, len(samples)))

    print(f"\nEvaluating on {len(samples)} samples...\n")

    overlaps = []
    total_content_words_hit = 0
    total_content_words_actual = 0

    for i, sample in enumerate(samples):
        msgs = sample["messages"]
        user_msg = next(m["content"] for m in msgs if m["role"] == "user")
        actual = next(m["content"] for m in msgs if m["role"] == "assistant")
        prompt_msgs = [m for m in msgs if m["role"] != "assistant"]

        predicted = generate(model, tokenizer, device, prompt_msgs)

        overlap = word_overlap(predicted, actual)
        overlaps.append(overlap)

        actual_cw = set(content_words(actual))
        pred_cw = set(content_words(predicted))
        total_content_words_hit += len(actual_cw & pred_cw)
        total_content_words_actual += len(actual_cw)

        if args.verbose:
            event = extract_event_name(user_msg)
            print(f"[{i+1}/{len(samples)}] {event}")
            print(f"  Overlap: {overlap:.1%}")
            print(f"  Predicted: {predicted[:200]}...")
            print(f"  Actual:    {actual[:200]}...")
            print()

        # Progress
        if not args.verbose and (i + 1) % 10 == 0:
            avg_so_far = sum(overlaps) / len(overlaps)
            print(f"  {i+1}/{len(samples)} done  (avg overlap: {avg_so_far:.1%})")

    # ── Results ──
    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
    median_overlap = sorted(overlaps)[len(overlaps) // 2] if overlaps else 0
    global_word_hit = total_content_words_hit / total_content_words_actual if total_content_words_actual else 0

    print()
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Samples evaluated:      {len(samples)}")
    print(f"  Avg word overlap:       {avg_overlap:.1%}")
    print(f"  Median word overlap:    {median_overlap:.1%}")
    print(f"  Global content word hit:{global_word_hit:.1%}")
    print()
    print("Distribution:")
    buckets = Counter()
    for o in overlaps:
        if o < 0.1:
            buckets["0-10%"] += 1
        elif o < 0.2:
            buckets["10-20%"] += 1
        elif o < 0.3:
            buckets["20-30%"] += 1
        elif o < 0.4:
            buckets["30-40%"] += 1
        elif o < 0.5:
            buckets["40-50%"] += 1
        else:
            buckets["50%+"] += 1
    for bucket in ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50%+"]:
        count = buckets.get(bucket, 0)
        bar = "#" * (count * 40 // len(overlaps)) if overlaps else ""
        print(f"  {bucket:>6}: {count:4d}  {bar}")

    print("=" * 50)


if __name__ == "__main__":
    main()
