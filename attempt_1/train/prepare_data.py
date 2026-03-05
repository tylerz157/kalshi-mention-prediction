#!/usr/bin/env python3
"""
prepare_data.py -- Parse 313 Trump transcripts and build raw instruction JSONL samples.

Sampling strategy
-----------------
For each Trump segment, create overlapping ~150-word target windows (stride 100 → ~50-word
overlap between adjacent samples). This gives dense coverage while exposing the model to
slightly different starting contexts for similar content.

Sentence integrity
------------------
- Target:  after collecting TARGET_WORDS, trim back to the last complete sentence
           (finds last . ! ? and cuts there — no mid-sentence targets).
- Context: after truncating to CONTEXT_CHARS, trim the start forward to the first
           clean speaker-segment boundary (first \\n\\n) so we never start mid-sentence.

Question-break bonus samples
-----------------------------
When a substantive interviewer question follows a Trump segment, an extra sample is
created whose context ends right after the question. This captures the Q→A training
signal where the question is the most prominent part of the context.

Output fields per sample
------------------------
  event_name          TITLE line from transcript
  source_file         filename (for tracing)
  approx_position     0.0–1.0 within the transcript
  words_before        exact word count of all speech before the cut point
  context_verbatim    Event title + last ~CONTEXT_CHARS of prior multi-speaker text
  target_text         ~150 Trump words trimmed to last complete sentence
  sample_type         'dense' | 'question_break'
  summary_placeholder blank — filled by add_summaries.py

Run from the train/ directory:
  python prepare_data.py
"""

import json
import random
import re
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
TRANSCRIPT_DIR = Path('../transcripts/senate_dems')
OUTPUT_DIR     = Path('data')

# ── tunable constants ─────────────────────────────────────────────────────────
TARGET_WORDS      = 150  # collect ~N Trump words per window, then trim to sentence
TRUMP_STRIDE      = 75   # stride between windows → ~75-word overlap
MIN_TARGET_WORDS  = 50   # discard targets shorter than this after sentence trimming
CONTEXT_CHARS     = 8000 # max chars of verbatim prior context (not counting title)
FILLER_MAX_WORDS  = 6    # non-Trump utterances ≤ this length are filler candidates
VAL_FRACTION      = 0.10
RANDOM_SEED       = 42

# ── parsing constants ─────────────────────────────────────────────────────────
SPEAKER_RE   = re.compile(r'^([A-Z][A-Za-z0-9 .,\'-]{1,40}?)\s*:\s*(.*)$')
NON_SPEAKERS = frozenset({'note', 'source', 'url', 'title', 'warning'})

QUESTION_STARTERS = frozenset({
    'what', 'how', 'when', 'why', 'where', 'who',
    'did', 'do', 'does', 'is', 'are', 'can', 'will',
    'would', 'should', 'could', 'have', 'has',
})

# Sentence end: .  !  ?  optionally followed by closing quote, then space or end-of-string
SENT_END_RE = re.compile(r'[.!?]["\']?(?=\s|$)')


# ── helpers ───────────────────────────────────────────────────────────────────

def is_trump_speaker(name: str) -> bool:
    return 'trump' in name.lower()


def is_filler(text: str) -> bool:
    """Return True if this non-Trump utterance is a back-channel filler response."""
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
    """
    Trim text to end at the last complete sentence.
    If no sentence boundary is found, return the text as-is (Trump sometimes trails off).
    """
    matches = list(SENT_END_RE.finditer(text))
    if not matches:
        return text.strip()
    return text[:matches[-1].end()].strip()


def trim_context_start(verbatim: str) -> str:
    """
    When verbatim context was truncated from the left, the first segment may be partial
    (starting mid-sentence). Trim forward to the first clean \\n\\n separator so the
    context always begins at the start of a speaker segment.
    Only applies if a \\n\\n appears within the first ~200 chars.
    """
    nn = verbatim.find('\n\n')
    if 0 < nn < 200:
        return verbatim[nn + 2:]
    return verbatim


def parse_transcript(path: Path) -> tuple[str, list[tuple[str, str]]]:
    """
    Return (event_name, [(speaker, text), ...]).
    Strips header (TITLE/URL/SOURCE/=== lines) and any leading Note block.
    All speakers preserved; [Inaudible] annotations kept as-is.
    """
    text = path.read_text(encoding='utf-8', errors='replace')
    lines = text.splitlines()

    event_name = path.stem
    if lines and lines[0].startswith('TITLE:'):
        event_name = lines[0].removeprefix('TITLE:').strip()

    # Find end of header
    body_start = 0
    for i, line in enumerate(lines):
        if line.startswith('==='):
            body_start = i + 1
            break

    # Skip optional leading Note block
    if body_start < len(lines) and lines[body_start].startswith('Note:'):
        while body_start < len(lines) and lines[body_start].strip():
            body_start += 1

    # Parse into (speaker, text) segments
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
    """
    Build context string: event title + last CONTEXT_CHARS of prior speech.

    up_to_idx: segments[:up_to_idx] are included in the prior speech.
    extra_words: if provided, these words from segments[up_to_idx] are appended before
                 truncation (used when the context window falls mid-segment).

    Context start is trimmed to a clean speaker-segment boundary after truncation.
    """
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


def words_in_segments(segments: list[tuple[str, str]], up_to_idx: int, extra_word_count: int = 0) -> int:
    """Count total words spoken across segments[:up_to_idx], plus any extra words."""
    return sum(len(txt.split()) for _, txt in segments[:up_to_idx]) + extra_word_count


def build_samples(
    event_name: str,
    segments: list[tuple[str, str]],
    source_file: str,
) -> list[dict]:
    samples: list[dict] = []
    # Track (approx_pos, target_prefix) to avoid duplicate question_break samples
    seen_targets: set[tuple[float, str]] = set()

    n_segs = max(len(segments) - 1, 1)

    for seg_idx, (spk, seg_text) in enumerate(segments):
        if not is_trump_speaker(spk):
            continue

        seg_words = seg_text.split()
        approx_pos = round(seg_idx / n_segs, 3)

        # ── dense sub-windows through this Trump segment ──────────────────────
        word_offset = 0
        while word_offset < len(seg_words):
            window_words = seg_words[word_offset : word_offset + TARGET_WORDS]

            # Trim to last complete sentence
            raw = ' '.join(window_words)
            target_text = trim_to_last_sentence(raw)

            if len(target_text.split()) >= MIN_TARGET_WORDS:
                # Context includes prior segments + words[0:word_offset] of this segment
                context = build_context(
                    event_name, segments, seg_idx,
                    extra_words=seg_words[:word_offset] if word_offset > 0 else None,
                )
                wb = words_in_segments(segments, seg_idx, word_offset)
                key = (approx_pos, target_text[:40])
                if key not in seen_targets:
                    seen_targets.add(key)
                    samples.append({
                        'event_name':          event_name,
                        'source_file':         source_file,
                        'approx_position':     approx_pos,
                        'words_before':        wb,
                        'context_verbatim':    context,
                        'target_text':         target_text,
                        'sample_type':         'dense',
                        'summary_placeholder': '',
                    })

            word_offset += TRUMP_STRIDE

        # ── question-break bonus sample ───────────────────────────────────────
        # If a substantive non-Trump utterance follows this segment, create a sample
        # whose context ends right after that question — high-signal Q→A training pair.
        next_q_idx = None
        for j in range(seg_idx + 1, len(segments)):
            nspk, ntxt = segments[j]
            if is_trump_speaker(nspk):
                break  # Trump spoke again before any question
            if not is_filler(ntxt):
                next_q_idx = j
                break

        if next_q_idx is not None:
            # Find Trump's next response after the question
            next_trump_idx = next(
                (j for j in range(next_q_idx + 1, len(segments))
                 if is_trump_speaker(segments[j][0])),
                None,
            )
            if next_trump_idx is not None:
                resp_words = segments[next_trump_idx][1].split()
                raw_resp = ' '.join(resp_words[:TARGET_WORDS])
                resp_target = trim_to_last_sentence(raw_resp)

                if len(resp_target.split()) >= MIN_TARGET_WORDS:
                    # Context = everything up to and including the question
                    context = build_context(event_name, segments, next_q_idx + 1)
                    resp_approx = round(next_trump_idx / n_segs, 3)
                    # words_before = all words up to and including the question segment
                    wb = words_in_segments(segments, next_q_idx + 1)
                    key = (resp_approx, resp_target[:40])
                    if key not in seen_targets:
                        seen_targets.add(key)
                        samples.append({
                            'event_name':          event_name,
                            'source_file':         source_file,
                            'approx_position':     resp_approx,
                            'words_before':        wb,
                            'context_verbatim':    context,
                            'target_text':         resp_target,
                            'sample_type':         'question_break',
                            'summary_placeholder': '',
                        })

    return samples


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    for path in transcript_files:
        try:
            event_name, segments = parse_transcript(path)
        except Exception as e:
            print(f"  ERROR {path.name}: {e}")
            errors += 1
            continue

        samples = build_samples(event_name, segments, path.name)
        if not samples:
            skipped += 1
            continue

        if path.name in val_names:
            val_samples.extend(samples)
        else:
            train_samples.extend(samples)

    train_out = OUTPUT_DIR / 'instruct_raw_train.jsonl'
    val_out   = OUTPUT_DIR / 'instruct_raw_val.jsonl'

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
    print(f"  Total:          {total}  (train {len(train_samples)}, val {len(val_samples)})")
    print(f"  Dense:          {dense_n}")
    print(f"  Question-break: {qbreak_n}")
    print(f"  Skipped:        {skipped} | Errors: {errors}")
    print(f"\nOutput: {train_out}, {val_out}")

    # Target length distribution
    all_samples = train_samples + val_samples
    lengths = sorted(len(s['target_text'].split()) for s in all_samples)
    n = len(lengths)
    if n:
        print(f"\nTarget word counts:")
        print(f"  Min {lengths[0]}  p10 {lengths[n//10]}  median {lengths[n//2]}"
              f"  p90 {lengths[9*n//10]}  max {lengths[-1]}")

    # Print one example of each type
    for label, stype in [('DENSE', 'dense'), ('QUESTION-BREAK', 'question_break')]:
        ex = next((s for s in train_samples if s['sample_type'] == stype), None)
        if ex:
            print(f"\n--- Example [{label}] ---")
            print(f"Event:    {ex['event_name'][:80]}")
            print(f"Position: {ex['approx_position']:.0%} into speech")
            print(f"Context tail:\n  ...{ex['context_verbatim'][-350:]}")
            print(f"Target ({len(ex['target_text'].split())} words):\n  {ex['target_text']}")


if __name__ == '__main__':
    main()
