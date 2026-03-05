#!/usr/bin/env python3
"""
add_summaries.py -- Enrich raw instruction samples with news_context:
  Top news headlines about Trump from the 3 days before the speech,
  fetched from The Guardian API and written directly (no AI summarization).

Cached per date so each unique date is only one API call (~100 calls total).

Resumable: on restart, reads existing output file and skips already-processed samples.

Usage:
  python add_summaries.py              # process train + val
  python add_summaries.py --dry-run   # process first 5 samples only (for testing)

Requirements:
  pip install requests
  GUARDIAN_API_KEY env var must be set.
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

# ── Paths ─────────────────────────────────────────────────────────────────────

INPUT_TRAIN  = Path('data/instruct_raw_train.jsonl')
INPUT_VAL    = Path('data/instruct_raw_val.jsonl')
OUTPUT_TRAIN = Path('data/instruct_enriched_train.jsonl')
OUTPUT_VAL   = Path('data/instruct_enriched_val.jsonl')

# ── Constants ─────────────────────────────────────────────────────────────────

GUARDIAN_BASE = 'https://content.guardianapis.com/search'
HEADLINES_COUNT = 10   # headlines to include per date
LOOKBACK_DAYS   = 3    # fetch news from this many days before the speech
API_DELAY       = 0.2  # seconds between API calls

# ── Date parsing ──────────────────────────────────────────────────────────────

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


def parse_date(event_name: str, source_file: str) -> datetime | None:
    # Try event_name: M.D.YY or MM.DD.YY
    m = re.search(r'\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b', event_name)
    if m:
        mon, day, yr = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = (2000 + yr) if yr < 100 else yr
        d = _try_date(year, mon, day)
        if d:
            return d

    # Try source_file: full month name + day + year
    m = re.search(
        r'-(' + '|'.join(MONTH_MAP) + r')-(\d{1,2})-(\d{4})\.txt',
        source_file, re.IGNORECASE
    )
    if m:
        mon = MONTH_MAP[m.group(1).lower()]
        day, year = int(m.group(2)), int(m.group(3))
        d = _try_date(year, mon, day)
        if d:
            return d

    # Try source_file: compact digits (MMDDYYYY, MMDDYY, MDDYY, MDYY)
    m = re.search(r'[-_](\d{4,8})(?:\.txt)?$', source_file)
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


# ── Guardian news fetcher ──────────────────────────────────────────────────────

def fetch_guardian_headlines(date: datetime, api_key: str) -> str:
    """
    Fetch top Trump-related headlines from the Guardian for the
    LOOKBACK_DAYS days before the speech date.
    Returns a newline-separated list of headlines, or empty string on failure.
    """
    from_date = (date - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    to_date   = (date - timedelta(days=1)).strftime('%Y-%m-%d')

    try:
        resp = requests.get(
            GUARDIAN_BASE,
            params={
                'q':            'trump',
                'from-date':    from_date,
                'to-date':      to_date,
                'api-key':      api_key,
                'show-fields':  'headline,sectionId',
                'page-size':    25,       # fetch extra so we have room to filter
                'order-by':     'relevance',
            },
            timeout=12,
            headers={'User-Agent': 'Mozilla/5.0 (research project, not commercial)'}
        )
        if resp.status_code != 200:
            print(f"[guardian {resp.status_code}]", end=' ')
            return ''

        results = resp.json().get('response', {}).get('results', [])
        headlines = []
        for r in results:
            # Skip non-news sections
            skip_sections = {
                'commentisfree', 'lifeandstyle', 'culture', 'sport',
                'australia-news', 'global-development', 'environment',
                'science', 'technology', 'food', 'travel', 'fashion',
            }
            if r.get('sectionId') in skip_sections:
                continue
            headline = r.get('fields', {}).get('headline') or r.get('webTitle', '')
            if headline:
                headlines.append(f"- {headline}")
            if len(headlines) >= HEADLINES_COUNT:
                break

        return '\n'.join(headlines)

    except Exception as e:
        print(f"[guardian fetch failed: {e}]", end=' ')
        return ''


# ── Resume support ─────────────────────────────────────────────────────────────

def load_done_keys(output_path: Path) -> set:
    done = set()
    if output_path.exists():
        for line in output_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                try:
                    s = json.loads(line)
                    done.add((s['source_file'], s['words_before']))
                except Exception:
                    pass
    return done


# ── Core enrichment loop ──────────────────────────────────────────────────────

def enrich_file(
    input_path: Path,
    output_path: Path,
    api_key: str,
    news_cache: dict,
    dry_run: bool = False,
):
    samples = [
        json.loads(l)
        for l in input_path.read_text(encoding='utf-8').splitlines()
        if l.strip()
    ]
    done_keys = load_done_keys(output_path)
    to_process = [s for s in samples if (s['source_file'], s['words_before']) not in done_keys]

    if dry_run:
        to_process = to_process[:5]

    print(
        f"\n{input_path.name}: {len(samples)} total, "
        f"{len(done_keys)} already done, {len(to_process)} to process"
    )
    if not to_process:
        print("  Nothing to do.")
        return

    api_calls = 0

    with open(output_path, 'a', encoding='utf-8') as out_f:
        for i, s in enumerate(to_process, 1):
            print(f"  [{i}/{len(to_process)}] {s['source_file'][:55]} @{s['words_before']}w", end=' ', flush=True)

            date = parse_date(s['event_name'], s['source_file'])
            if date:
                date_key = date.strftime('%Y-%m-%d')
                if date_key not in news_cache:
                    print(f"guardian({date_key})...", end=' ', flush=True)
                    news_cache[date_key] = fetch_guardian_headlines(date, api_key)
                    api_calls += 1
                    time.sleep(API_DELAY)
                    print(f"done", end=' ', flush=True)
                else:
                    print(f"cached({date_key})", end=' ', flush=True)
                news_context = news_cache[date_key]
            else:
                news_context = ''
                print("no-date", end=' ', flush=True)

            print(flush=True)

            enriched = {**s, 'news_context': news_context}
            out_f.write(json.dumps(enriched) + '\n')
            out_f.flush()

    print(f"\n  Done. API calls this run: {api_calls}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dry_run = '--dry-run' in sys.argv

    api_key = os.environ.get('GUARDIAN_API_KEY')
    if not api_key:
        print("ERROR: GUARDIAN_API_KEY environment variable not set.")
        print("  Set it with: $env:GUARDIAN_API_KEY='your-key-here'")
        sys.exit(1)

    news_cache: dict = {}

    if dry_run:
        print("DRY RUN mode: processing first 5 samples per file only.")

    for input_path, output_path in [
        (INPUT_TRAIN, OUTPUT_TRAIN),
        (INPUT_VAL,   OUTPUT_VAL),
    ]:
        if not input_path.exists():
            print(f"Skipping {input_path} (not found)")
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enrich_file(input_path, output_path, api_key, news_cache, dry_run=dry_run)

    print("\nAll done.")
    print(f"  Unique dates cached: {len(news_cache)}")


if __name__ == '__main__':
    main()
