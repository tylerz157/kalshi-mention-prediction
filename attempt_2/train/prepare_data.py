#!/usr/bin/env python3
"""
prepare_data.py (attempt_2) -- Build keyword YES/NO training samples.

For each transcript position, given a 150-word look-ahead window and
Kalshi candidate words for that event:
  - YES: Kalshi words found in the window (ordered by first appearance)
         + non-Kalshi content keywords (stop-word filtered)
  - NO:  Kalshi candidate words NOT found in the window (alphabetical)

Output fields per sample:
  event_name, source_file, approx_position, words_before,
  context_verbatim, target_text (YES/NO format), kalshi_candidates,
  sample_type ('dense' | 'question_break'), summary_placeholder

Run from attempt_2/train/:
  python prepare_data.py
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
TRANSCRIPT_DIR       = Path('../../transcripts/senate_dems')
SHARED_DATA_DIR      = Path('../../train/data')  # shared Kalshi data
OUTPUT_DIR           = Path('data')

KALSHI_MAP_FILE      = SHARED_DATA_DIR / 'transcript_kalshi_map.json'
KALSHI_BY_EVENT_FILE = SHARED_DATA_DIR / 'kalshi_words_by_event.json'

# ── tunable constants ─────────────────────────────────────────────────────────
TARGET_WORDS      = 150   # collect ~N Trump words per window for keyword extraction
TRUMP_STRIDE      = 75    # stride between windows
MIN_TARGET_WORDS  = 50    # discard targets shorter than this
CONTEXT_CHARS     = 8000  # max chars of verbatim prior context
FILLER_MAX_WORDS  = 6     # non-Trump utterances <= this length are filler
VAL_FRACTION      = 0.10
RANDOM_SEED       = 42

# ── parsing constants ─────────────────────────────────────────────────────────
SPEAKER_RE   = re.compile(r'^([A-Z][A-Za-z0-9 .,\'-]{1,40}?)\s*:\s*(.*)$')
NON_SPEAKERS = frozenset({'note', 'source', 'url', 'title', 'warning'})
WORD_RE      = re.compile(r'[a-z]+')

QUESTION_STARTERS = frozenset({
    'what', 'how', 'when', 'why', 'where', 'who',
    'did', 'do', 'does', 'is', 'are', 'can', 'will',
    'would', 'should', 'could', 'have', 'has',
})

SENT_END_RE = re.compile(r'[.!?]["\']?(?=\s|$)')

# ── stop words ───────────────────────────────────────────────────────────────
STOP_WORDS = frozenset({
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
    "really", "want", "tell", "people", "gonna", "dont",
    "ive", "thats", "theyre", "youre", "weve", "hes", "shes",
    "theyve", "youve", "didnt", "doesnt", "isnt", "arent",
    "wasnt", "werent", "wont", "wouldnt", "couldnt", "shouldnt",
    "havent", "hasnt", "hadnt", "one", "two", "yeah", "yes", "okay",
    "oh", "hey", "let", "see", "put", "new", "old", "big", "great",
    "good", "bad", "first", "last", "long", "little", "own", "same",
    "another", "around", "most", "every", "down", "made", "only",
    "more", "time", "happen", "happened", "happens",
})


# ── Kalshi data loading ──────────────────────────────────────────────────────

def load_kalshi_data() -> tuple[dict, list[str]]:
    """Load transcript->Kalshi map and compute default candidate list.
    Returns (transcript_map, default_candidates).
    """
    transcript_map = {}
    if KALSHI_MAP_FILE.exists():
        with open(KALSHI_MAP_FILE, encoding='utf-8') as f:
            transcript_map = json.load(f)

    by_event = {}
    if KALSHI_BY_EVENT_FILE.exists():
        with open(KALSHI_BY_EVENT_FILE, encoding='utf-8') as f:
            by_event = json.load(f)

    # Default candidates: top 25 most common words across all events
    word_counts = Counter()
    for words in by_event.values():
        word_counts.update(words)
    default_candidates = [w for w, _ in word_counts.most_common(25)]

    return transcript_map, default_candidates


def get_kalshi_candidates(source_file: str, transcript_map: dict,
                          default_candidates: list[str]) -> list[str]:
    """Get Kalshi candidate words for a specific transcript."""
    info = transcript_map.get(source_file, {})
    kalshi_events = info.get('kalshi_events', [])
    if kalshi_events:
        words = set()
        for event in kalshi_events:
            for market in event.get('markets', []):
                raw = market.get('word', '')
                for variant in raw.split(' / '):
                    v = variant.strip().lower()
                    if v:
                        words.add(v)
        return sorted(words)
    return default_candidates


# ── keyword extraction ────────────────────────────────────────────────────────

def find_kalshi_matches(text: str, candidates: list[str]) -> list[tuple[str, int]]:
    """Find which Kalshi candidate words appear in text with word-boundary matching.
    Returns list of (word, first_char_position) sorted by position.
    """
    text_lower = text.lower()
    matches = []
    for word in candidates:
        # Use word boundaries to avoid substring matches (e.g. "ai" in "said")
        pattern = r'\b' + re.escape(word) + r'\b'
        m = re.search(pattern, text_lower)
        if m:
            matches.append((word, m.start()))
    matches.sort(key=lambda x: x[1])
    return matches


def extract_content_keywords(text: str) -> list[str]:
    """Extract content keywords (no stop words, >2 chars), ordered by first appearance."""
    seen = set()
    keywords = []
    for word in WORD_RE.findall(text.lower()):
        if word not in STOP_WORDS and len(word) > 2 and word not in seen:
            seen.add(word)
            keywords.append(word)
    return keywords


def build_keyword_target(raw_text: str, kalshi_candidates: list[str]) -> str:
    """Build YES/NO keyword target from raw text and Kalshi candidates.
    Only Kalshi market words — no extra content keywords.
    YES: Kalshi words found (ordered by first appearance).
    NO: Kalshi words not found (alphabetical).
    """
    kalshi_yes = find_kalshi_matches(raw_text, kalshi_candidates)
    kalshi_yes_words = [w for w, _ in kalshi_yes]
    kalshi_yes_set = set(kalshi_yes_words)

    kalshi_no = sorted(w for w in kalshi_candidates if w not in kalshi_yes_set)

    yes_str = ', '.join(kalshi_yes_words) if kalshi_yes_words else '(none)'
    no_str = ', '.join(kalshi_no) if kalshi_no else '(none)'

    return f"YES: {yes_str}\nNO: {no_str}"


# ── transcript parsing (reused from attempt_1) ───────────────────────────────

def is_trump_speaker(name: str) -> bool:
    return 'trump' in name.lower()


def is_filler(text: str) -> bool:
    text = text.strip()
    words = text.split()
    if not words:
        return True
    if '?' in text:
        return False
    if len(words) > FILLER_MAX_WORDS:
        return False
    first = words[0].lower().rstrip('.,!-')
    return first not in QUESTION_STARTERS


def trim_to_last_sentence(text: str) -> str:
    matches = list(SENT_END_RE.finditer(text))
    if not matches:
        return text.strip()
    return text[:matches[-1].end()].strip()


def trim_context_start(verbatim: str) -> str:
    nn = verbatim.find('\n\n')
    if 0 < nn < 200:
        return verbatim[nn + 2:]
    return verbatim


def parse_transcript(path: Path) -> tuple[str, list[tuple[str, str]]]:
    text = path.read_text(encoding='utf-8', errors='replace')
    lines = text.splitlines()

    event_name = path.stem
    if lines and lines[0].startswith('TITLE:'):
        event_name = lines[0].removeprefix('TITLE:').strip()

    body_start = 0
    for i, line in enumerate(lines):
        if line.startswith('==='):
            body_start = i + 1
            break

    if body_start < len(lines) and lines[body_start].startswith('Note:'):
        while body_start < len(lines) and lines[body_start].strip():
            body_start += 1

    segments: list[tuple[str, str]] = []
    current_speaker: str | None = None
    current_parts: list[str] = []

    def flush():
        if current_speaker and current_parts:
            combined = ' '.join(current_parts).strip()
            if combined:
                segments.append((current_speaker, combined))

    for line in lines[body_start:]:
        m = SPEAKER_RE.match(line)
        if m:
            candidate = m.group(1).strip()
            rest = m.group(2).strip()
            if candidate.lower() not in NON_SPEAKERS:
                flush()
                current_speaker = candidate
                current_parts = [rest] if rest else []
                continue
        stripped = line.strip()
        if stripped and current_speaker is not None:
            current_parts.append(stripped)

    flush()
    return event_name, segments


def build_context(
    event_name: str,
    segments: list[tuple[str, str]],
    up_to_idx: int,
    extra_words: list[str] | None = None,
) -> str:
    prior_parts = [f"{spk}: {txt}" for spk, txt in segments[:up_to_idx] if txt.strip()]
    if extra_words:
        spk = segments[up_to_idx][0]
        prior_parts.append(f"{spk}: {' '.join(extra_words)}")

    full_prior = '\n\n'.join(prior_parts)

    if len(full_prior) > CONTEXT_CHARS:
        verbatim = full_prior[-CONTEXT_CHARS:]
        verbatim = trim_context_start(verbatim)
    else:
        verbatim = full_prior

    return f"Event: {event_name}\n\n{verbatim}"


def words_in_segments(segments, up_to_idx, extra_word_count=0):
    return sum(len(txt.split()) for _, txt in segments[:up_to_idx]) + extra_word_count


# ── sample building ──────────────────────────────────────────────────────────

def build_samples(
    event_name: str,
    segments: list[tuple[str, str]],
    source_file: str,
    kalshi_candidates: list[str],
) -> list[dict]:
    samples: list[dict] = []
    seen_targets: set[tuple[float, str]] = set()
    n_segs = max(len(segments) - 1, 1)

    for seg_idx, (spk, seg_text) in enumerate(segments):
        if not is_trump_speaker(spk):
            continue

        seg_words = seg_text.split()
        approx_pos = round(seg_idx / n_segs, 3)

        # ── dense sub-windows ──
        word_offset = 0
        while word_offset < len(seg_words):
            window_words = seg_words[word_offset : word_offset + TARGET_WORDS]
            raw = ' '.join(window_words)
            trimmed = trim_to_last_sentence(raw)

            if len(trimmed.split()) >= MIN_TARGET_WORDS:
                context = build_context(
                    event_name, segments, seg_idx,
                    extra_words=seg_words[:word_offset] if word_offset > 0 else None,
                )
                wb = words_in_segments(segments, seg_idx, word_offset)
                target_text = build_keyword_target(trimmed, kalshi_candidates)
                key = (approx_pos, wb)
                if key not in seen_targets:
                    seen_targets.add(key)
                    samples.append({
                        'event_name':          event_name,
                        'source_file':         source_file,
                        'approx_position':     approx_pos,
                        'words_before':        wb,
                        'context_verbatim':    context,
                        'target_text':         target_text,
                        'raw_target':          trimmed,
                        'kalshi_candidates':   kalshi_candidates,
                        'sample_type':         'dense',
                        'summary_placeholder': '',
                    })

            word_offset += TRUMP_STRIDE

        # ── question-break bonus sample ──
        next_q_idx = None
        for j in range(seg_idx + 1, len(segments)):
            nspk, ntxt = segments[j]
            if is_trump_speaker(nspk):
                break
            if not is_filler(ntxt):
                next_q_idx = j
                break

        if next_q_idx is not None:
            next_trump_idx = next(
                (j for j in range(next_q_idx + 1, len(segments))
                 if is_trump_speaker(segments[j][0])),
                None,
            )
            if next_trump_idx is not None:
                resp_words = segments[next_trump_idx][1].split()
                raw_resp = ' '.join(resp_words[:TARGET_WORDS])
                resp_trimmed = trim_to_last_sentence(raw_resp)

                if len(resp_trimmed.split()) >= MIN_TARGET_WORDS:
                    context = build_context(event_name, segments, next_q_idx + 1)
                    resp_approx = round(next_trump_idx / n_segs, 3)
                    wb = words_in_segments(segments, next_q_idx + 1)
                    target_text = build_keyword_target(resp_trimmed, kalshi_candidates)
                    raw_target = resp_trimmed
                    key = (resp_approx, wb)
                    if key not in seen_targets:
                        seen_targets.add(key)
                        samples.append({
                            'event_name':          event_name,
                            'source_file':         source_file,
                            'approx_position':     resp_approx,
                            'words_before':        wb,
                            'context_verbatim':    context,
                            'target_text':         target_text,
                            'raw_target':          raw_target,
                            'kalshi_candidates':   kalshi_candidates,
                            'sample_type':         'question_break',
                            'summary_placeholder': '',
                        })

    return samples


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transcript_map, default_candidates = load_kalshi_data()
    print(f"Loaded Kalshi map: {len(transcript_map)} transcripts mapped")
    print(f"Default candidates ({len(default_candidates)}): {', '.join(default_candidates[:10])}...")

    transcript_files = sorted(TRANSCRIPT_DIR.glob('*.txt'))
    print(f"Found {len(transcript_files)} transcripts")

    random.seed(RANDOM_SEED)
    shuffled = list(transcript_files)
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * VAL_FRACTION))
    val_names   = {p.name for p in shuffled[:n_val]}
    train_names = {p.name for p in shuffled[n_val:]}
    print(f"Train: {len(train_names)} transcripts | Val: {len(val_names)} transcripts")

    train_samples: list[dict] = []
    val_samples:   list[dict] = []
    skipped = errors = 0
    matched_kalshi = 0

    for path in transcript_files:
        try:
            event_name, segments = parse_transcript(path)
        except Exception as e:
            print(f"  ERROR {path.name}: {e}")
            errors += 1
            continue

        candidates = get_kalshi_candidates(path.name, transcript_map, default_candidates)
        if candidates != default_candidates:
            matched_kalshi += 1

        samples = build_samples(event_name, segments, path.name, candidates)
        if not samples:
            skipped += 1
            continue

        if path.name in val_names:
            val_samples.extend(samples)
        else:
            train_samples.extend(samples)

    train_out = OUTPUT_DIR / 'keyword_train.jsonl'
    val_out   = OUTPUT_DIR / 'keyword_val.jsonl'

    with open(train_out, 'w', encoding='utf-8') as f:
        for s in train_samples:
            f.write(json.dumps(s) + '\n')

    with open(val_out, 'w', encoding='utf-8') as f:
        for s in val_samples:
            f.write(json.dumps(s) + '\n')

    total    = len(train_samples) + len(val_samples)
    dense_n  = sum(1 for s in train_samples + val_samples if s['sample_type'] == 'dense')
    qbreak_n = sum(1 for s in train_samples + val_samples if s['sample_type'] == 'question_break')

    print(f"\nResults:")
    print(f"  Total:              {total}  (train {len(train_samples)}, val {len(val_samples)})")
    print(f"  Dense:              {dense_n}")
    print(f"  Question-break:     {qbreak_n}")
    print(f"  Matched to Kalshi:  {matched_kalshi} transcripts")
    print(f"  Using defaults:     {len(transcript_files) - matched_kalshi - errors - skipped}")
    print(f"  Skipped:            {skipped} | Errors: {errors}")
    print(f"\nOutput: {train_out}, {val_out}")

    # Show keyword stats
    all_samples = train_samples + val_samples
    yes_counts = []
    no_counts = []
    for s in all_samples:
        lines = s['target_text'].split('\n')
        yes_line = lines[0].removeprefix('YES: ')
        no_line = lines[1].removeprefix('NO: ') if len(lines) > 1 else ''
        yes_n = 0 if yes_line == '(none)' else len(yes_line.split(', '))
        no_n = 0 if no_line == '(none)' else len(no_line.split(', '))
        yes_counts.append(yes_n)
        no_counts.append(no_n)

    print(f"\nYES keyword counts: min={min(yes_counts)} median={sorted(yes_counts)[len(yes_counts)//2]} max={max(yes_counts)}")
    print(f"NO keyword counts:  min={min(no_counts)} median={sorted(no_counts)[len(no_counts)//2]} max={max(no_counts)}")

    # Print examples
    for label, stype in [('DENSE', 'dense'), ('QUESTION-BREAK', 'question_break')]:
        ex = next((s for s in train_samples if s['sample_type'] == stype), None)
        if ex:
            print(f"\n--- Example [{label}] ---")
            print(f"Event:      {ex['event_name'][:80]}")
            print(f"Position:   {ex['approx_position']:.0%} into speech")
            print(f"Candidates: {', '.join(ex['kalshi_candidates'][:10])}...")
            print(f"Context tail:\n  ...{ex['context_verbatim'][-300:]}")
            print(f"Target:\n  {ex['target_text']}")


if __name__ == '__main__':
    main()
