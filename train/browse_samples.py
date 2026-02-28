"""
browse_samples.py -- Randomly sample through training data for visual inspection.
Press Enter for next sample, q+Enter to quit.

Usage:
  python browse_samples.py              # browse train set
  python browse_samples.py val          # browse val set
  python browse_samples.py question_break  # filter to question_break samples only
"""

import json
import random
import sys
from pathlib import Path

SEP = "-" * 72

def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]

def show(s):
    print(SEP)
    wb = s.get('words_before', '?')
    print(f"  TYPE:     {s['sample_type']}   |   POSITION: {s['approx_position']:.0%} into speech  |  WORDS BEFORE: {wb}")
    print(f"  EVENT:    {s['event_name'][:80]}")
    print(f"  FILE:     {s['source_file']}")
    print(SEP)
    print("  CONTEXT (full):")
    for line in s['context_verbatim'].splitlines():
        print(f"    {line}")
    print()
    print(f"  TARGET ({len(s['target_text'].split())} words):")
    for line in s['target_text'].splitlines():
        print(f"    {line}")
    print()

def main():
    args = sys.argv[1:]
    split_arg    = 'val' if 'val' in args else 'train'
    type_filter  = next((a for a in args if a in ('dense', 'question_break')), None)

    file = Path('data') / f'instruct_raw_{split_arg}.jsonl'
    if not file.exists():
        print(f"File not found: {file}")
        return

    samples = load(file)
    if type_filter:
        samples = [s for s in samples if s['sample_type'] == type_filter]

    print(f"Loaded {len(samples)} samples from {file.name}"
          + (f" (filtered to '{type_filter}')" if type_filter else ""))
    print("Press Enter for next sample, q+Enter to quit.\n")

    random.shuffle(samples)
    idx = 0

    while True:
        show(samples[idx % len(samples)])
        idx += 1
        try:
            key = input("  [Enter = next | q = quit] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if key == 'q':
            break

if __name__ == '__main__':
    main()
