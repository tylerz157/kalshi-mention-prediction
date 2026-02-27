"""Generate Fort Bragg transcript vs market chart."""
import ast, json, os, sys, io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def pn(v):
    try: return ast.literal_eval(v)
    except: return None

EVENT        = 'KXTRUMPMENTION-26FEB14'
TRANSCRIPT   = 'transcripts/senate_dems/transcript-president-trump-addresses-military-families-at-fort-bragg-north-carolina-2132026.txt'
SPEECH_START = pd.Timestamp('2026-02-13 18:20:00', tz='UTC')
SPEECH_END   = pd.Timestamp('2026-02-13 19:37:00', tz='UTC')
WPM          = 75  # calibrated: 5792 words over 77 min

FOCUS = {
    'Hottest': 'yes', 'Tariff': 'yes', 'Biden': 'yes', 'Venezuela': 'yes',
    'Trillion': 'yes', 'Nuclear': 'yes', 'Eight War': 'yes',
    'National Security': 'no', 'Hegseth': 'no', 'Middle East': 'no', 'Nobel': 'no',
}

with open(TRANSCRIPT, encoding='utf-8') as f:
    raw = f.read()
lines = raw.split('\n')
ti = next(i for i, l in enumerate(lines) if '='*10 in l) + 1
transcript = '\n'.join(lines[ti:]).strip()
tl = transcript.lower()
print(f'Transcript: {len(transcript.split())} words')

with open('data/kalshi/trump_mention_markets.json') as f:
    markets = json.load(f)
mdict = {m['yes_sub_title']: m for m in markets if m['event_ticker'] == EVENT}

def w2ts(n):
    return SPEECH_START + pd.Timedelta(seconds=(n / WPM) * 60)

word_ts = {}
for word in FOCUS:
    s = word.split('/')[0].strip().lower()
    p = tl.find(s)
    if p == -1:
        for alt in word.split('/')[1:]:
            p = tl.find(alt.strip().lower())
            if p != -1:
                break
    if p != -1:
        idx = len(tl[:p].split())
        word_ts[word] = w2ts(idx)
        print(f'  {word}: word #{idx} -> {word_ts[word].strftime("%H:%M")} UTC')
    else:
        word_ts[word] = None
        print(f'  {word}: NOT IN TRANSCRIPT')


def load_df(ticker):
    path = f'data/kalshi/{ticker}_candlesticks.csv'
    if not os.path.exists(path):
        return None
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
    window = df['ts'].between(SPEECH_START - pd.Timedelta('30min'),
                               SPEECH_END   + pd.Timedelta('15min'))
    return df[window].sort_values('ts')


fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(17, 10),
    gridspec_kw={'height_ratios': [3, 1]},
    sharex=True
)
fig.suptitle(
    'Fort Bragg Military Families  |  Feb 13 2026  (18:20–19:37 UTC)\n'
    'Market prices vs transcript word timing\n'
    'Dotted vertical line = estimated time word first spoken (from transcript)',
    fontsize=11, y=0.99
)

cyg = plt.cm.Greens(np.linspace(0.45, 0.9, sum(1 for r in FOCUS.values() if r == 'yes')))
cyr = plt.cm.Reds(  np.linspace(0.45, 0.9, sum(1 for r in FOCUS.values() if r == 'no')))
iy = ir = 0
vols = []

for word, result in FOCUS.items():
    meta = mdict.get(word)
    if not meta:
        continue
    df = load_df(meta['ticker'])
    if df is None or df.empty:
        continue

    c  = cyg[iy] if result == 'yes' else cyr[ir]
    ls = '-'     if result == 'yes' else '--'
    lw = 2.0     if result == 'yes' else 1.5
    if result == 'yes':
        iy += 1
    else:
        ir += 1

    tag   = 'YES' if result == 'yes' else 'NO '
    label = f"{tag}: {word}"
    ax1.plot(df['ts'], df['mid'], color=c, lw=lw, ls=ls, label=label)

    est = word_ts.get(word)
    if est:
        ax1.axvline(est, color=c, lw=1.8, alpha=0.55, ls=':')
        ax1.text(est, 102, word[:9], fontsize=6, color=c,
                 rotation=90, va='top', ha='right')

    vols.append(df.set_index('ts')[['vol']])

ax1.axvspan(SPEECH_START, SPEECH_END, alpha=0.04, color='royalblue')
ax1.axvline(SPEECH_START, color='royalblue', lw=1.5, ls='--', alpha=0.6, label='Speech start/end')
ax1.axvline(SPEECH_END,   color='royalblue', lw=1.5, ls='--', alpha=0.6)
ax1.axhline(50, color='gray', ls=':', alpha=0.35, lw=0.8)
ax1.set_ylim(-2, 107)
ax1.set_ylabel('YES Price  (cents = % probability)', fontsize=10)
ax1.legend(fontsize=7.5, loc='upper left', ncol=2, framealpha=0.8)
ax1.set_title(
    'GREEN solid = resolved YES    RED dashed = resolved NO    '
    'Dotted vertical = word position in transcript (estimated timing)',
    fontsize=8.5
)

if vols:
    vc = pd.concat(vols).resample('1min').sum()
    ax2.bar(vc.index, vc['vol'], width=pd.Timedelta('55s'), color='steelblue', alpha=0.7)
    ax2.axvline(SPEECH_START, color='royalblue', lw=1.5, ls='--', alpha=0.6)
    ax2.axvline(SPEECH_END,   color='royalblue', lw=1.5, ls='--', alpha=0.6)
    ax2.set_ylabel('Volume / min', fontsize=9)

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone.utc))
ax2.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')
ax2.set_xlabel('Time UTC  (Feb 13 2026)', fontsize=10)

plt.tight_layout()
out = 'fort_bragg_transcript_vs_market.png'
plt.savefig(out, dpi=140, bbox_inches='tight')
print(f'\nSaved: {out}')
