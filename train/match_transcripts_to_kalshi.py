#!/usr/bin/env python3
"""
match_transcripts_to_kalshi.py -- Match transcript files to Kalshi events by date.

For each transcript, finds all Kalshi events on that same date and stores the
associated markets (words + results). Saves a map to:
  train/data/transcript_kalshi_map.json

This map is used by:
  - add_summaries.py  : to know which Kalshi words were being tracked
  - validate.py       : to check if model predictions matched market outcomes

Run from the train/ directory:
  python match_transcripts_to_kalshi.py
"""

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

TRANSCRIPT_DIR   = Path('../transcripts/senate_dems')
EVENTS_FILE      = Path('../past_kalshi_markets/trump_events_checkpoint.json')
MARKETS_FILE     = Path('../past_kalshi_markets/trump_mention_markets.json')
OUTPUT_MAP       = Path('data/transcript_kalshi_map.json')

# ── Kalshi date parsing ────────────────────────────────────────────────────────

KALSHI_MONTH = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
    'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
    'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12,
}

def parse_kalshi_date(event_ticker: str) -> datetime | None:
    """
    Parse date from Kalshi event ticker.
    Format: SERIES-YYMONDD[suffix]
    e.g. KXTRUMPMENTION-26FEB28      -> 2026-02-28
         KXTRUMPMENTIONB-25NOV11B    -> 2025-11-11
         KXTRUMPMENTION-25OCT10-2    -> 2025-10-10
         KXTRUMPMENTION-26JAN222     -> 2026-01-22
    Strategy: take the second dash-separated token, then extract YYMONDD with regex.
    """
    parts = event_ticker.split('-')
    if len(parts) < 2:
        return None
    date_str = parts[1]  # e.g. '26FEB28', '25NOV11B', '26JAN222'
    m = re.match(r'(\d{2})([A-Z]{3})(\d{1,2})', date_str)
    if not m:
        return None
    yy, mon_str, dd = int(m.group(1)), m.group(2), int(m.group(3))
    mon = KALSHI_MONTH.get(mon_str)
    if not mon:
        return None
    try:
        return datetime(2000 + yy, mon, dd)
    except ValueError:
        return None


# ── Transcript date parsing ────────────────────────────────────────────────────
# Reuse the same logic as add_summaries.py

MONTH_MAP = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12,
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7,
    'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
}


def _try_date(year, month, day):
    try:
        return datetime(year, month, day)
    except ValueError:
        return None


def parse_transcript_date(filename: str) -> datetime | None:
    """Parse date from transcript filename (same logic as add_summaries.py)."""
    # Try full month name: -june-18-2025.txt
    m = re.search(
        r'-(' + '|'.join(MONTH_MAP) + r')-(\d{1,2})-(\d{4})\.txt',
        filename, re.IGNORECASE
    )
    if m:
        mon = MONTH_MAP[m.group(1).lower()]
        d = _try_date(int(m.group(3)), mon, int(m.group(2)))
        if d:
            return d

    # Try compact digits at end: MMDDYYYY, MMDDYY, MDDYY, MDYY
    m = re.search(r'[-_](\d{4,8})(?:\.txt)?$', filename)
    if m:
        digits = m.group(1)
        n = len(digits)
        candidates = []
        if n == 8:
            candidates.append((int(digits[4:]), int(digits[:2]), int(digits[2:4])))
        elif n == 6:
            candidates.append((2000 + int(digits[4:]), int(digits[:2]), int(digits[2:4])))
        elif n == 5:
            candidates.append((2000 + int(digits[3:]), int(digits[0]), int(digits[1:3])))
            candidates.append((2000 + int(digits[3:]), int(digits[:2]), int(digits[2])))
        elif n == 4:
            candidates.append((2000 + int(digits[2:]), int(digits[0]), int(digits[1])))
        for year, mon, day in candidates:
            d = _try_date(year, mon, day)
            if d and 1 <= mon <= 12 and 1 <= day <= 31 and 2020 <= year <= 2030:
                return d

    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load Kalshi events
    with open(EVENTS_FILE) as f:
        checkpoint = json.load(f)
    events = checkpoint['trump_events']
    print(f"Loaded {len(events)} Kalshi events")

    # Load all markets and group by event_ticker
    with open(MARKETS_FILE) as f:
        all_markets = json.load(f)
    markets_by_event = defaultdict(list)
    for m in all_markets:
        markets_by_event[m['event_ticker']].append({
            'ticker':        m['ticker'],
            'word':          m.get('yes_sub_title', ''),
            'result':        m.get('result', ''),
            'status':        m.get('status', ''),
            'volume':        m.get('volume', 0),
            'last_price':    m.get('last_price', 0),
        })
    print(f"Loaded {len(all_markets)} markets across {len(markets_by_event)} events")

    # Build date -> list of Kalshi events
    kalshi_by_date: dict[str, list] = defaultdict(list)
    unparsed_events = []
    for e in events:
        d = parse_kalshi_date(e['event_ticker'])
        if d:
            date_key = d.strftime('%Y-%m-%d')
            kalshi_by_date[date_key].append({
                'event_ticker': e['event_ticker'],
                'title':        e.get('title', ''),
                'markets':      markets_by_event.get(e['event_ticker'], []),
            })
        else:
            unparsed_events.append(e['event_ticker'])

    print(f"Parsed dates for {len(events) - len(unparsed_events)}/{len(events)} Kalshi events")
    if unparsed_events:
        print(f"  Could not parse: {unparsed_events}")

    # Load all transcripts and parse their dates
    transcript_files = sorted(TRANSCRIPT_DIR.glob('*.txt'))
    print(f"\nFound {len(transcript_files)} transcript files")

    transcript_map = {}  # filename -> {date, kalshi_events}
    matched = 0
    unmatched_transcripts = []
    no_date_transcripts = []

    for tf in transcript_files:
        d = parse_transcript_date(tf.name)
        if d is None:
            no_date_transcripts.append(tf.name)
            continue

        date_key = d.strftime('%Y-%m-%d')
        kalshi_events = kalshi_by_date.get(date_key, [])

        transcript_map[tf.name] = {
            'date':          date_key,
            'kalshi_events': kalshi_events,
        }
        if kalshi_events:
            matched += 1
        else:
            unmatched_transcripts.append((tf.name, date_key))

    # Save map
    OUTPUT_MAP.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MAP, 'w', encoding='utf-8') as f:
        json.dump(transcript_map, f, indent=2)

    # Report
    total_dated = len(transcript_map)
    print(f"\nResults:")
    print(f"  Transcripts with parseable date: {total_dated}/{len(transcript_files)}")
    print(f"  Matched to Kalshi event:         {matched}/{total_dated}")
    print(f"  No Kalshi event on that date:    {total_dated - matched}")
    if no_date_transcripts:
        print(f"  Could not parse date from:       {len(no_date_transcripts)} files")

    print(f"\nMatched transcripts:")
    for fname, info in sorted(transcript_map.items(), key=lambda x: x[1]['date']):
        if info['kalshi_events']:
            event_strs = ', '.join(e['event_ticker'] for e in info['kalshi_events'])
            n_markets = sum(len(e['markets']) for e in info['kalshi_events'])
            print(f"  {info['date']}  {fname[:55]:<55}  -> {event_strs}  ({n_markets} markets)")

    print(f"\nUnmatched dates (transcript exists, no Kalshi event):")
    for fname, date_key in sorted(unmatched_transcripts, key=lambda x: x[1]):
        print(f"  {date_key}  {fname[:70]}")

    if no_date_transcripts:
        print(f"\nCould not parse date from:")
        for fname in no_date_transcripts:
            print(f"  {fname}")

    print(f"\nMap saved to {OUTPUT_MAP}")

    # Word-level summary for matched events
    all_words = []
    for info in transcript_map.values():
        for event in info['kalshi_events']:
            for m in event['markets']:
                all_words.append(m['word'])
    if all_words:
        from collections import Counter
        top = Counter(all_words).most_common(20)
        print(f"\nTop 20 most-tracked Kalshi words across matched events:")
        for word, count in top:
            print(f"  {count:3d}x  {word}")


if __name__ == '__main__':
    main()
