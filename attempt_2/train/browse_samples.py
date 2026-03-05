#!/usr/bin/env python3
"""
browse_samples.py -- Browse keyword training samples interactively.

Usage:
  python browse_samples.py                # random val samples
  python browse_samples.py --train        # random train samples
  python browse_samples.py --n 5          # show 5 samples
  python browse_samples.py --kalshi-only  # only show samples with Kalshi YES hits
  python browse_samples.py --openai       # show the full OpenAI chat format
"""

import argparse
import json
import random
import sys
from pathlib import Path

VAL_FILE   = Path('data/keyword_val.jsonl')
TRAIN_FILE = Path('data/keyword_train.jsonl')
OPENAI_VAL = Path('data/openai_val.jsonl')
OPENAI_TRAIN = Path('data/openai_train.jsonl')


def parse_target(target_text):
    lines = target_text.split('\n')
    yes_line = lines[0].removeprefix('YES: ') if lines else ''
    no_line = lines[1].removeprefix('NO: ') if len(lines) > 1 else ''
    yes_words = [w.strip() for w in yes_line.split(', ')] if yes_line and yes_line != '(none)' else []
    no_words = [w.strip() for w in no_line.split(', ')] if no_line and no_line != '(none)' else []
    return yes_words, no_words


def show_sample(s, idx, total, show_openai=False, openai_sample=None):
    yes_words, no_words = parse_target(s['target_text'])
    candidates = set(s.get('kalshi_candidates', []))
    kalshi_yes = [w for w in yes_words if w in candidates]
    kalshi_no = [w for w in no_words if w in candidates]
    extra_yes = [w for w in yes_words if w not in candidates]

    print(f"\n{'=' * 70}")
    print(f"  SAMPLE {idx}/{total}  |  {s['sample_type'].upper()}  |  {s['approx_position']:.0%} into speech")
    print(f"{'=' * 70}")
    print(f"\n  Event: {s['event_name']}")
    print(f"  File:  {s['source_file']}")
    print(f"  Words before cut point: {s['words_before']}")

    print(f"\n  --- KALSHI CANDIDATES ({len(s.get('kalshi_candidates', []))}) ---")
    print(f"  {', '.join(s.get('kalshi_candidates', []))}")

    print(f"\n  --- CONTEXT (last 500 chars of {len(s['context_verbatim'])} total) ---")
    print(f"  ...{s['context_verbatim'][-500:]}")

    print(f"\n  --- TARGET ---")
    print(f"  Kalshi YES ({len(kalshi_yes)}): {', '.join(kalshi_yes) if kalshi_yes else '(none)'}")
    print(f"  Kalshi NO  ({len(kalshi_no)}): {', '.join(kalshi_no) if kalshi_no else '(none)'}")
    print(f"  Extra keywords ({len(extra_yes)}): {', '.join(extra_yes[:20])}{'...' if len(extra_yes) > 20 else ''}")

    # Show raw target text so user can verify keywords
    raw = s.get('raw_target', '')
    if raw:
        print(f"\n  --- ORIGINAL TEXT (what Trump actually said next) ---")
        print(f"  {raw}")

    if show_openai and openai_sample:
        print(f"\n  --- FULL OPENAI FORMAT ---")
        for msg in openai_sample['messages']:
            role = msg['role'].upper()
            content = msg['content']
            print(f"\n  [{role}]")
            for line in content.split('\n'):
                print(f"  {line}")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Use train set instead of val')
    parser.add_argument('--n', type=int, default=3, help='Number of samples to show')
    parser.add_argument('--kalshi-only', action='store_true', help='Only samples with Kalshi YES hits')
    parser.add_argument('--openai', action='store_true', help='Show full OpenAI chat format')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    args = parser.parse_args()

    raw_file = TRAIN_FILE if args.train else VAL_FILE
    openai_file = OPENAI_TRAIN if args.train else OPENAI_VAL

    samples = [json.loads(l) for l in raw_file.read_text(encoding='utf-8').splitlines() if l.strip()]

    openai_samples = None
    if args.openai and openai_file.exists():
        openai_samples = [json.loads(l) for l in openai_file.read_text(encoding='utf-8').splitlines() if l.strip()]

    if args.kalshi_only:
        filtered = []
        for i, s in enumerate(samples):
            yes_words, _ = parse_target(s['target_text'])
            candidates = set(s.get('kalshi_candidates', []))
            if any(w in candidates for w in yes_words):
                filtered.append((i, s))
        print(f"Filtered to {len(filtered)}/{len(samples)} samples with Kalshi YES hits")
        indices = [i for i, _ in filtered]
        samples_pool = [(i, s) for i, s in filtered]
    else:
        samples_pool = list(enumerate(samples))

    if args.seed is not None:
        random.seed(args.seed)
    random.shuffle(samples_pool)

    n = min(args.n, len(samples_pool))
    for pick in range(n):
        orig_idx, s = samples_pool[pick]
        openai_s = openai_samples[orig_idx] if openai_samples and orig_idx < len(openai_samples) else None
        show_sample(s, pick + 1, n, show_openai=args.openai, openai_sample=openai_s)

        if pick < n - 1:
            try:
                if sys.stdin.isatty():
                    input("  Press Enter for next sample (Ctrl+C to quit)...")
            except (KeyboardInterrupt, EOFError):
                print("\n  Done.")
                return


if __name__ == '__main__':
    main()
