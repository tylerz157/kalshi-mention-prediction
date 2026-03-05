#!/usr/bin/env python3
"""
kalshi_words.py -- Extract and normalize Kalshi market words from market data.

Outputs:
  train/data/kalshi_word_list.json       - Global flat list of all unique normalized words
  train/data/kalshi_words_by_event.json  - Per-event ticker -> list of normalized words

Run from the train/ directory:
  python kalshi_words.py
"""

import json
import re
from collections import Counter
from pathlib import Path

MARKETS_FILE = Path('../past_kalshi_markets/trump_mention_markets.json')
OUTPUT_GLOBAL = Path('data/kalshi_word_list.json')
OUTPUT_BY_EVENT = Path('data/kalshi_words_by_event.json')


def normalize_word(raw: str) -> list[str]:
    """
    Normalize a yes_sub_title into matchable word(s).

    Handles:
      - Slash-separated variants: "Crypto / Bitcoin" -> ["crypto", "bitcoin"]
      - "The State of The / Our Union is Strong" -> ["state of the union is strong", "state of our union is strong"]
      - Simple words: "Tariff" -> ["tariff"]
    """
    raw = raw.strip()
    if not raw:
        return []

    # Split on " / " to get variants
    variants = [v.strip() for v in raw.split(' / ')]

    results = []
    for v in variants:
        v = v.lower().strip()
        if v:
            results.append(v)

    return results


def main():
    with open(MARKETS_FILE, encoding='utf-8') as f:
        markets = json.load(f)

    print(f"Loaded {len(markets)} markets")

    # Global word set
    all_words = set()
    # Per-event word lists
    by_event: dict[str, set] = {}

    for m in markets:
        event = m.get('event_ticker', '')
        raw_word = m.get('yes_sub_title', '')
        normalized = normalize_word(raw_word)

        all_words.update(normalized)

        if event not in by_event:
            by_event[event] = set()
        by_event[event].update(normalized)

    # Convert sets to sorted lists
    global_list = sorted(all_words)
    by_event_list = {k: sorted(v) for k, v in sorted(by_event.items())}

    # Save
    OUTPUT_GLOBAL.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_GLOBAL, 'w', encoding='utf-8') as f:
        json.dump(global_list, f, indent=2)

    with open(OUTPUT_BY_EVENT, 'w', encoding='utf-8') as f:
        json.dump(by_event_list, f, indent=2)

    print(f"\nGlobal unique words: {len(global_list)}")
    print(f"Events with words:   {len(by_event_list)}")

    # Show word frequency
    word_counts = Counter()
    for words in by_event.values():
        word_counts.update(words)

    print(f"\nTop 30 most common Kalshi words (across events):")
    for word, count in word_counts.most_common(30):
        print(f"  {count:3d}x  {word}")

    print(f"\nWords per event:")
    sizes = [len(v) for v in by_event_list.values()]
    print(f"  Min: {min(sizes)}  Max: {max(sizes)}  Avg: {sum(sizes)/len(sizes):.1f}")

    print(f"\nSaved to {OUTPUT_GLOBAL} and {OUTPUT_BY_EVENT}")


if __name__ == '__main__':
    main()
