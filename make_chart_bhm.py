"""Black History Month (Feb 18 2026) transcript vs market chart with excerpts."""
import ast, json, os, re, io, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from datetime import timezone
from textwrap import fill

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

EVENT        = 'KXTRUMPMENTION-26FEB19'
TRANSCRIPT   = 'transcripts/timestamped/black_history_month_2026_02_19.txt'
VIDEO_START  = pd.Timestamp('2026-02-18 20:12:00', tz='UTC')  # calibrated from market data
SPEECH_END   = pd.Timestamp('2026-02-18 20:59:00', tz='UTC')  # ~47 min

# Words to show: (kalshi_title, result, [search_terms])
FOCUS = {
    'Democrat':        ('yes', ['democrat']),
    'Election':        ('yes', ['election']),
    'Hottest':         ('yes', ['hottest']),
    'Biden':           ('yes', ['biden']),
    'Stock Market':    ('yes', ['stock market']),
    'ICE / National Guard': ('yes', ['national guard']),
    'Crime / Criminal': ('yes', ['criminal', 'crime ']),
    'Bad Bunny':       ('no',  ['bad bunny']),
    'Epstein':         ('no',  ['epstein']),
    'Supreme Court':   ('no',  ['supreme court']),
    'Crypto / Bitcoin': ('no', ['crypto', 'bitcoin']),
}

# ── Parse transcript ───────────────────────────────────────────────────────────
with open(TRANSCRIPT, encoding='utf-8') as f:
    raw = f.read()

body = raw.split('=' * 10)[-1].strip()

# Single pass: find every timestamp marker, track current speaker
# Matches: "Speaker Name (MM:SS):" and continuation "(MM:SS)"
token_re = re.compile(
    r'(?:^([A-Za-z][A-Za-z .0-9]*?)\s*)?\((\d{1,2}):(\d{2})\)\s*:?',
    re.MULTILINE
)
segments = []
tokens = list(token_re.finditer(body))
current_speaker = 'Unknown'
for i, tok in enumerate(tokens):
    speaker_name = tok.group(1)
    mins = int(tok.group(2))
    secs = int(tok.group(3))
    elapsed = mins * 60 + secs
    if speaker_name:
        current_speaker = speaker_name.strip()
    # Text runs until the next token
    start = tok.end()
    end   = tokens[i+1].start() if i+1 < len(tokens) else len(body)
    text  = body[start:end].strip()
    segments.append((elapsed, current_speaker, text))

segments.sort(key=lambda x: x[0])

print(f'Parsed {len(segments)} transcript segments')
for ts, sp, tx in segments[:5]:
    print(f'  {ts//60:02d}:{ts%60:02d} {sp[:20]}: {tx[:60]}...')

def ts_to_utc(elapsed_secs):
    return VIDEO_START + pd.Timedelta(seconds=elapsed_secs)

def find_word_in_transcript(search_terms):
    """Return (elapsed_secs, excerpt_200_chars) for first Trump mention of any search term."""
    for elapsed, speaker, text in segments:
        if 'trump' not in speaker.lower():
            continue
        tl = text.lower()
        for term in search_terms:
            if term in tl:
                # Find position of term in text
                pos = tl.find(term)
                # Get context: up to 200 chars before this point in full speech
                # Build a rolling window of recent Trump text up to this moment
                return elapsed, text
    return None, None

def get_excerpt_before(elapsed_secs, word, n_chars=350):
    """Get n_chars of text spoken (by anyone) just before elapsed_secs."""
    # Collect all text up to this point
    all_text = []
    for es, sp, tx in segments:
        if es <= elapsed_secs:
            all_text.append(tx.replace('\n', ' '))
    combined = ' '.join(all_text)
    # Return last n_chars
    excerpt = combined[-n_chars:].strip()
    # Highlight the keyword
    return excerpt

# ── Find word timestamps ───────────────────────────────────────────────────────
word_info = {}
print('\nWord timing (Trump utterances only):')
for word, (result, terms) in FOCUS.items():
    elapsed, text = find_word_in_transcript(terms)
    if elapsed is not None:
        utc = ts_to_utc(elapsed)
        excerpt = get_excerpt_before(elapsed, word, n_chars=300)
        word_info[word] = {'result': result, 'elapsed': elapsed, 'utc': utc, 'excerpt': excerpt}
        print(f'  {word} ({result.upper()}): {elapsed//60:02d}:{elapsed%60:02d} -> {utc.strftime("%H:%M")} UTC')
        print(f'    ...{excerpt[-100:]}')
    else:
        word_info[word] = {'result': result, 'elapsed': None, 'utc': None, 'excerpt': None}
        print(f'  {word} ({result.upper()}): NOT FOUND IN TRUMP SEGMENTS')

# ── Load market data ───────────────────────────────────────────────────────────
with open('data/kalshi/trump_mention_markets.json') as f:
    markets = json.load(f)
mdict = {m['yes_sub_title']: m for m in markets if m['event_ticker'] == EVENT}

def pn(v):
    try: return ast.literal_eval(v)
    except: return None

def load_df(ticker):
    path = f'data/kalshi/{ticker}_candlesticks.csv'
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    df['ts']  = pd.to_datetime(df['end_period_ts'].astype(int), unit='s', utc=True)
    df['ask'] = df['yes_ask'].apply(pn).apply(lambda d: d.get('close') if isinstance(d, dict) else None)
    df['bid'] = df['yes_bid'].apply(pn).apply(lambda d: d.get('close') if isinstance(d, dict) else None)
    df['pc']  = df['price'].apply(pn).apply(lambda d: d.get('close') if isinstance(d, dict) else None)
    df['mid'] = df.apply(
        lambda r: r['pc'] if r['pc'] else (
            (r['ask'] + r['bid']) / 2 if r['ask'] and r['bid'] else None
        ), axis=1
    )
    df['vol'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    window = df['ts'].between(
        VIDEO_START - pd.Timedelta('30min'),
        SPEECH_END  + pd.Timedelta('15min')
    )
    return df[window].sort_values('ts')

# ── Build figure ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
gs  = fig.add_gridspec(3, 1, height_ratios=[4, 1, 3], hspace=0.08)
ax1 = fig.add_subplot(gs[0])   # price
ax2 = fig.add_subplot(gs[1], sharex=ax1)   # volume
ax3 = fig.add_subplot(gs[2])   # transcript excerpts

fig.suptitle(
    'White House Black History Month Event  |  Feb 18 2026\n'
    'Kalshi market prices vs timestamped transcript\n'
    'Dotted vertical = confirmed word spoken (from transcript)',
    fontsize=11, y=0.995
)

n_yes = sum(1 for v in FOCUS.values() if v[0] == 'yes')
n_no  = sum(1 for v in FOCUS.values() if v[0] == 'no')
cyg = plt.cm.Greens(np.linspace(0.4, 0.95, n_yes))
cyr = plt.cm.Reds(  np.linspace(0.4, 0.95, n_no))
iy = ir = 0
vols = []

for word, (result, terms) in FOCUS.items():
    meta = mdict.get(word)
    if not meta: continue
    df = load_df(meta['ticker'])
    if df is None or df.empty: continue

    c  = cyg[iy] if result == 'yes' else cyr[ir]
    ls = '-'     if result == 'yes' else '--'
    lw = 2.0     if result == 'yes' else 1.5
    if result == 'yes': iy += 1
    else:               ir += 1

    tag   = 'YES' if result == 'yes' else 'NO '
    label = f'{tag}: {word}'
    ax1.plot(df['ts'], df['mid'], color=c, lw=lw, ls=ls, label=label)

    info = word_info.get(word, {})
    if info.get('utc'):
        ax1.axvline(info['utc'], color=c, lw=1.8, alpha=0.6, ls=':')
        ax1.text(info['utc'], 103, word[:10], fontsize=5.5, color=c,
                 rotation=90, va='top', ha='right')

    vols.append(df.set_index('ts')[['vol']])

ax1.axvspan(VIDEO_START, SPEECH_END, alpha=0.04, color='royalblue')
ax1.axvline(VIDEO_START, color='royalblue', lw=1.5, ls='--', alpha=0.6, label='Speech start/end')
ax1.axvline(SPEECH_END,  color='royalblue', lw=1.5, ls='--', alpha=0.6)
ax1.axhline(50, color='gray', ls=':', alpha=0.3, lw=0.8)
ax1.set_ylim(-2, 112)
ax1.set_ylabel('YES Price  (cents)', fontsize=10)
ax1.legend(fontsize=7, loc='upper left', ncol=2, framealpha=0.85)
ax1.set_title(
    'GREEN solid = resolved YES    RED dashed = resolved NO\n'
    'Dotted verticals = exact time word spoken per transcript (video_start = 20:12 UTC, calibrated)',
    fontsize=8
)

if vols:
    vc = pd.concat(vols).resample('1min').sum()
    ax2.bar(vc.index, vc['vol'], width=pd.Timedelta('55s'), color='steelblue', alpha=0.7)
    ax2.axvline(VIDEO_START, color='royalblue', lw=1.5, ls='--', alpha=0.6)
    ax2.axvline(SPEECH_END,  color='royalblue', lw=1.5, ls='--', alpha=0.6)
    ax2.set_ylabel('Vol/min', fontsize=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone.utc))
ax2.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')
ax2.set_xlabel('Time UTC  (Feb 18 2026)', fontsize=10)

# ── Transcript excerpt panel ───────────────────────────────────────────────────
ax3.axis('off')
ax3.set_title('Transcript excerpts — text spoken just before each key word (Trump segments only)',
              fontsize=8, loc='left')

# Show YES words found in transcript with excerpts
found_words = [(w, info) for w, info in word_info.items()
               if info['utc'] is not None and info['result'] == 'yes']
found_words.sort(key=lambda x: x[1]['elapsed'])

col_width = 1.0 / max(len(found_words), 1)
for col_i, (word, info) in enumerate(found_words):
    x = col_i * col_width + 0.01
    elapsed = info['elapsed']
    utc_str = info['utc'].strftime('%H:%M UTC') if info['utc'] else '?'
    mm = elapsed // 60
    ss = elapsed % 60
    excerpt = info['excerpt'] or ''
    # Wrap text
    wrapped = fill(f'...{excerpt[-220:]}', width=28)
    header = f'{word}\n{mm:02d}:{ss:02d} in video / {utc_str}'
    ax3.text(x, 0.95, header, transform=ax3.transAxes,
             fontsize=6.5, va='top', ha='left', fontweight='bold',
             color='darkgreen')
    ax3.text(x, 0.80, wrapped, transform=ax3.transAxes,
             fontsize=5.5, va='top', ha='left', color='#333333',
             fontfamily='monospace')

# Also show NO words not found
not_found = [(w, info) for w, info in word_info.items()
             if info['utc'] is None and info['result'] == 'no']
if not_found:
    nf_str = 'NOT SPOKEN (resolved NO): ' + ', '.join(w for w, _ in not_found)
    ax3.text(0.0, 0.02, fill(nf_str, width=120), transform=ax3.transAxes,
             fontsize=6.5, va='bottom', ha='left', color='darkred')

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = 'bhm_transcript_vs_market.png'
plt.savefig(out, dpi=140, bbox_inches='tight')
print(f'\nSaved: {out}')

# ── Print full excerpts to console ────────────────────────────────────────────
print('\n' + '='*70)
print('FULL TRANSCRIPT EXCERPTS (300 chars before each word)')
print('='*70)
for word, info in sorted(word_info.items(), key=lambda x: x[1]['elapsed'] or 9999):
    if info['utc'] is None:
        continue
    elapsed = info['elapsed']
    print(f'\n--- {word} ({info["result"].upper()}) ---')
    print(f'    Video: {elapsed//60:02d}:{elapsed%60:02d}  |  UTC: {info["utc"].strftime("%H:%M:%S")}')
    print(f'    Excerpt:')
    print(fill(info['excerpt'], width=70, initial_indent='    ', subsequent_indent='    '))
