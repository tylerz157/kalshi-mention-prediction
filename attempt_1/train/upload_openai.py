#!/usr/bin/env python3
"""
upload_openai.py -- Format enriched samples into OpenAI chat format,
upload train/val files, and launch a gpt-4o-mini fine-tuning job.

Usage:
  python upload_openai.py            # format, upload, and start job
  python upload_openai.py --dry-run  # format only, print 2 examples, no upload

Requirements:
  pip install openai
  OPENAI_API_KEY env var must be set.
"""

import json
import os
import re
import sys
from pathlib import Path

import openai

# ── Paths ─────────────────────────────────────────────────────────────────────

INPUT_TRAIN  = Path('data/instruct_enriched_train.jsonl')
INPUT_VAL    = Path('data/instruct_enriched_val.jsonl')
OUTPUT_TRAIN = Path('data/openai_train.jsonl')
OUTPUT_VAL   = Path('data/openai_val.jsonl')

# ── Model ─────────────────────────────────────────────────────────────────────

FINETUNE_MODEL = 'gpt-4o-mini-2024-07-18'
N_EPOCHS       = 1

# ── Format ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are predicting what Donald Trump will say next in a live speech. "
    "Given recent news and the verbatim speech so far, predict what he says next."
)


def clean_event_name(event_name: str) -> str:
    """Strip 'TRANSCRIPT: ' prefix and trailing date from event name."""
    name = re.sub(r'^TRANSCRIPT:\s*', '', event_name).strip()
    # Remove trailing date patterns: ", 10.21.25" / " - October 27, 2025" / " 08.15.25"
    name = re.sub(r'[\s,\-]+(\d{1,2}[./]\d{1,2}[./]\d{2,4})\s*$', '', name).strip()
    name = re.sub(
        r'[\s,\-]+(january|february|march|april|may|june|july|august|september|october|november|december)'
        r'\s+\d{1,2},?\s*\d{4}\s*$', '', name, flags=re.IGNORECASE
    ).strip()
    return name


def format_sample(s: dict) -> dict:
    """Convert one enriched sample into OpenAI chat format."""
    parts = []

    event = clean_event_name(s['event_name'])
    words = s.get('words_before', '?')
    parts.append(f"Event: {event}")
    parts.append(f"Words spoken so far: {words}")

    if s.get('news_context'):
        parts.append(f"\nRecent news:\n{s['news_context']}")

    # Strip transcript header lines (Event:/URL:/SOURCE: metadata) from verbatim context
    context = re.sub(r'^(Event|URL|SOURCE|TITLE):.*\n?', '', s['context_verbatim'], flags=re.MULTILINE)
    context = context.lstrip('\n')
    parts.append(f"\nSpeech so far:\n{context}")

    parts.append("\nWhat does Trump say next?")

    user_content = '\n'.join(parts)

    return {
        'messages': [
            {'role': 'system',    'content': SYSTEM_PROMPT},
            {'role': 'user',      'content': user_content},
            {'role': 'assistant', 'content': s['target_text']},
        ]
    }


# ── Format and write ──────────────────────────────────────────────────────────

def format_file(input_path: Path, output_path: Path) -> int:
    samples = [
        json.loads(l)
        for l in input_path.read_text(encoding='utf-8').splitlines()
        if l.strip()
    ]
    with open(output_path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(format_sample(s)) + '\n')
    return len(samples)


# ── Token estimate ────────────────────────────────────────────────────────────

def estimate_tokens(path: Path) -> int:
    """Rough token estimate: chars / 4."""
    return path.stat().st_size // 4


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dry_run = '--dry-run' in sys.argv

    # Format both files
    print("Formatting samples...")
    for input_path, output_path in [
        (INPUT_TRAIN, OUTPUT_TRAIN),
        (INPUT_VAL,   OUTPUT_VAL),
    ]:
        n = format_file(input_path, output_path)
        tok = estimate_tokens(output_path)
        print(f"  {output_path.name}: {n} samples, ~{tok:,} tokens (~${tok/1e6*3:.2f} training cost)")

    if dry_run:
        print("\nDRY RUN -- showing first 2 formatted train samples:\n")
        samples = [
            json.loads(l)
            for l in OUTPUT_TRAIN.read_text(encoding='utf-8').splitlines()[:2]
            if l.strip()
        ]
        for i, s in enumerate(samples, 1):
            print(f"--- Sample {i} ---")
            for msg in s['messages']:
                role = msg['role'].upper()
                content = msg['content']
                print(f"[{role}]\n{content[:500]}{'...' if len(content) > 500 else ''}\n")
        return

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key)

    # Upload files
    print("\nUploading files to OpenAI...")
    train_file = client.files.create(
        file=open(OUTPUT_TRAIN, 'rb'),
        purpose='fine-tune'
    )
    print(f"  Train file ID: {train_file.id}")

    val_file = client.files.create(
        file=open(OUTPUT_VAL, 'rb'),
        purpose='fine-tune'
    )
    print(f"  Val file ID:   {val_file.id}")

    # Launch fine-tuning job
    print("\nLaunching fine-tuning job...")
    job = client.fine_tuning.jobs.create(
        training_file=train_file.id,
        validation_file=val_file.id,
        model=FINETUNE_MODEL,
        hyperparameters={'n_epochs': N_EPOCHS},
    )

    print(f"\nJob launched!")
    print(f"  Job ID:  {job.id}")
    print(f"  Status:  {job.status}")
    print(f"\nMonitor at: https://platform.openai.com/finetune")
    print(f"Or check status: openai api fine_tuning.jobs.retrieve -i {job.id}")

    # Save job ID for later
    info = {
        'job_id':        job.id,
        'train_file_id': train_file.id,
        'val_file_id':   val_file.id,
        'model':         FINETUNE_MODEL,
        'n_epochs':      N_EPOCHS,
    }
    Path('data/finetune_job.json').write_text(json.dumps(info, indent=2))
    print(f"\nJob info saved to data/finetune_job.json")


if __name__ == '__main__':
    main()
