#!/usr/bin/env python3
"""Quick inspection of keyword samples -- shows hard evidence the pipeline works."""
import json
from pathlib import Path

val_file = Path('data/keyword_val.jsonl')
samples = [json.loads(l) for l in val_file.read_text(encoding='utf-8').splitlines() if l.strip()]

# Find samples with actual Kalshi YES matches
kalshi_yes_samples = []
kalshi_no_only = []
for s in samples:
    lines = s['target_text'].split('\n')
    yes_line = lines[0].removeprefix('YES: ')
    candidates = set(s['kalshi_candidates'])
    yes_words = [w.strip() for w in yes_line.split(', ')] if yes_line != '(none)' else []
    kalshi_hits = [w for w in yes_words if w in candidates]
    if kalshi_hits:
        kalshi_yes_samples.append((s, kalshi_hits, yes_words))
    else:
        kalshi_no_only.append(s)

print("=" * 60)
print("KEYWORD PIPELINE EVIDENCE REPORT")
print("=" * 60)
print(f"Total val samples:           {len(samples)}")
print(f"Samples with Kalshi YES:     {len(kalshi_yes_samples)} ({len(kalshi_yes_samples)*100//len(samples)}%)")
print(f"Samples with all-Kalshi NO:  {len(kalshi_no_only)} ({len(kalshi_no_only)*100//len(samples)}%)")

# YES keyword count distribution
yes_counts = []
for s in samples:
    lines = s['target_text'].split('\n')
    yes_line = lines[0].removeprefix('YES: ')
    n = 0 if yes_line == '(none)' else len(yes_line.split(', '))
    yes_counts.append(n)

print(f"\nYES keyword count distribution:")
for bucket, lo, hi in [("0", 0, 0), ("1-5", 1, 5), ("6-15", 6, 15),
                        ("16-30", 16, 30), ("31-50", 31, 50), ("51+", 51, 999)]:
    count = sum(1 for c in yes_counts if lo <= c <= hi)
    bar = "#" * (count * 40 // len(yes_counts))
    print(f"  {bucket:>5}: {count:4d}  {bar}")

# Show 5 examples with Kalshi YES hits
print("\n" + "=" * 60)
print("EXAMPLE SAMPLES WITH KALSHI WORD HITS")
print("=" * 60)
for i, (s, hits, all_yes) in enumerate(kalshi_yes_samples[:5]):
    lines = s['target_text'].split('\n')
    no_line = lines[1].removeprefix('NO: ') if len(lines) > 1 else ''
    print(f"\n--- Sample {i+1} ---")
    print(f"Event:      {s['event_name'][:70]}")
    print(f"Position:   {s['approx_position']:.0%} | Type: {s['sample_type']}")
    print(f"Kalshi YES: {', '.join(hits)}")
    print(f"Kalshi NO:  {no_line[:120]}")
    print(f"All YES:    {', '.join(all_yes[:12])}{'...' if len(all_yes) > 12 else ''}")
    # Show last 200 chars of context for flavor
    ctx_tail = s['context_verbatim'][-200:]
    print(f"Context:    ...{ctx_tail}")

# Show 2 examples with ALL Kalshi NO (model should learn restraint)
print("\n" + "=" * 60)
print("EXAMPLE SAMPLES WITH ALL KALSHI WORDS = NO (restraint training)")
print("=" * 60)
for i, s in enumerate(kalshi_no_only[:2]):
    lines = s['target_text'].split('\n')
    yes_line = lines[0].removeprefix('YES: ')
    no_line = lines[1].removeprefix('NO: ') if len(lines) > 1 else ''
    print(f"\n--- Sample {i+1} ---")
    print(f"Event:    {s['event_name'][:70]}")
    print(f"YES:      {yes_line[:120]}")
    print(f"NO:       {no_line[:120]}")

# Kalshi word frequency in YES across all samples
print("\n" + "=" * 60)
print("MOST COMMON KALSHI WORDS IN YES (across all val samples)")
print("=" * 60)
from collections import Counter
kalshi_word_freq = Counter()
for s, hits, _ in kalshi_yes_samples:
    kalshi_word_freq.update(hits)
for word, count in kalshi_word_freq.most_common(20):
    pct = count * 100 // len(samples)
    print(f"  {count:4d} ({pct:2d}%)  {word}")
