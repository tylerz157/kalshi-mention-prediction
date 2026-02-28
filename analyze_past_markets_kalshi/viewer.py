"""Interactive transcript + market odds viewer."""
import ast, json, os, re
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide", page_title="Kalshi Transcript Viewer", page_icon="📈")

# ── Config ────────────────────────────────────────────────────────────────────
EVENT       = 'KXTRUMPMENTION-26FEB19'
TRANSCRIPT  = 'transcripts/timestamped/black_history_month_2026_02_19.txt'
VIDEO_START = pd.Timestamp('2026-02-18 20:12:00', tz='UTC')
SPEECH_END  = pd.Timestamp('2026-02-18 20:59:00', tz='UTC')

FOCUS = {
    'Democrat':           ['democrat'],
    'Election':           ['election'],
    'Hottest':            ['hottest'],
    'Biden':              ['biden'],
    'Stock Market':       ['stock market'],
    'ICE / National Guard': ['national guard'],
    'Crime / Criminal':   ['criminal', 'crime '],
    'Bad Bunny':          ['bad bunny'],
    'Epstein':            ['epstein'],
    'Supreme Court':      ['supreme court'],
    'Crypto / Bitcoin':   ['crypto', 'bitcoin'],
}

RESULT = {
    'Democrat': 'yes', 'Election': 'yes', 'Hottest': 'yes',
    'Biden': 'yes', 'Stock Market': 'yes', 'ICE / National Guard': 'yes',
    'Crime / Criminal': 'yes',
    'Bad Bunny': 'no', 'Epstein': 'no', 'Supreme Court': 'no',
    'Crypto / Bitcoin': 'no',
}

# Colors per word (consistent across panels)
WORD_COLORS = {
    'Democrat':           '#2ca02c',
    'Election':           '#17becf',
    'Hottest':            '#ff7f0e',
    'Biden':              '#9467bd',
    'Stock Market':       '#1f77b4',
    'ICE / National Guard': '#8c564b',
    'Crime / Criminal':   '#e377c2',
    'Bad Bunny':          '#d62728',
    'Epstein':            '#c49c94',
    'Supreme Court':      '#f7b6d2',
    'Crypto / Bitcoin':   '#bcbd22',
}

# ── Load & parse transcript ───────────────────────────────────────────────────
@st.cache_data
def load_transcript():
    with open(TRANSCRIPT, encoding='utf-8') as f:
        raw = f.read()
    body = raw.split('=' * 10)[-1].strip()
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
        start = tok.end()
        end   = tokens[i+1].start() if i+1 < len(tokens) else len(body)
        text  = body[start:end].strip()
        if text:
            segments.append({
                'elapsed': elapsed,
                'utc': VIDEO_START + pd.Timedelta(seconds=elapsed),
                'speaker': current_speaker,
                'text': text,
            })
    return sorted(segments, key=lambda x: x['elapsed'])

@st.cache_data
def load_markets():
    with open('data/kalshi/trump_mention_markets.json') as f:
        markets = json.load(f)
    return {m['yes_sub_title']: m for m in markets if m['event_ticker'] == EVENT}

def pn(v):
    try: return ast.literal_eval(v)
    except: return None

@st.cache_data
def load_candles(ticker):
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
    window = df['ts'].between(
        VIDEO_START - pd.Timedelta('25min'),
        SPEECH_END  + pd.Timedelta('15min')
    )
    return df[window].sort_values('ts').reset_index(drop=True)

def find_word_timestamp(segments, terms):
    """Return elapsed_secs of first Trump mention of any term."""
    for seg in segments:
        if 'trump' not in seg['speaker'].lower():
            continue
        tl = seg['text'].lower()
        for term in terms:
            if term in tl:
                return seg['elapsed'], seg['utc']
    return None, None

# ── Load data ─────────────────────────────────────────────────────────────────
segments = load_transcript()
mdict    = load_markets()

word_ts = {}  # word -> (elapsed, utc)
for word, terms in FOCUS.items():
    elapsed, utc = find_word_timestamp(segments, terms)
    word_ts[word] = (elapsed, utc)

# ── Build transcript HTML ─────────────────────────────────────────────────────
def build_transcript_html(selected_word):
    terms = FOCUS.get(selected_word, [])
    color = WORD_COLORS.get(selected_word, '#ff7f0e')

    trump_bg    = '#1a1a2e'
    other_bg    = '#16213e'
    speaker_clr = '#e0e0e0'

    rows = []
    for seg in segments:
        is_trump = 'trump' in seg['speaker'].lower()
        bg = trump_bg if is_trump else other_bg
        border = f'border-left: 3px solid {color};' if is_trump else ''
        mm = seg['elapsed'] // 60
        ss = seg['elapsed'] % 60
        utc_str = seg['utc'].strftime('%H:%M')

        text = seg['text']
        # Highlight selected word
        for term in terms:
            pat = re.compile(re.escape(term), re.IGNORECASE)
            text = pat.sub(
                f'<mark style="background:{color};color:#000;border-radius:3px;'
                f'padding:1px 3px;font-weight:bold">{term.upper()}</mark>',
                text
            )

        # Dim non-Trump segments when viewing a word
        opacity = '1.0' if is_trump else '0.55'

        rows.append(f'''
<div style="background:{bg};{border}padding:8px 10px;margin:3px 0;
            border-radius:4px;opacity:{opacity}">
  <span style="color:#888;font-size:11px;font-family:monospace">
    {mm:02d}:{ss:02d} | {utc_str} UTC
  </span>
  <span style="color:{color if is_trump else speaker_clr};
               font-size:12px;font-weight:{'bold' if is_trump else 'normal'};
               margin-left:8px">{seg["speaker"]}</span>
  <div style="color:#d0d0d0;font-size:13px;margin-top:4px;line-height:1.6">
    {text}
  </div>
</div>''')

    return '\n'.join(rows)

# ── Price chart ───────────────────────────────────────────────────────────────
def build_price_chart(selected_word):
    meta = mdict.get(selected_word)
    if not meta:
        return None
    df = load_candles(meta['ticker'])
    if df is None or df.empty:
        return None

    result   = RESULT.get(selected_word, '?')
    color    = WORD_COLORS.get(selected_word, '#ff7f0e')
    elapsed, utc = word_ts.get(selected_word, (None, None))

    fig = go.Figure()

    # Bid/ask band
    ask_vals = df['ask'].fillna(method='ffill')
    bid_vals = df['bid'].fillna(method='ffill')
    fig.add_trace(go.Scatter(
        x=pd.concat([df['ts'], df['ts'].iloc[::-1]]),
        y=pd.concat([ask_vals, bid_vals.iloc[::-1]]),
        fill='toself', fillcolor='rgba(100,100,200,0.12)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Bid/Ask spread', hoverinfo='skip'
    ))

    # Mid price
    fig.add_trace(go.Scatter(
        x=df['ts'], y=df['mid'],
        mode='lines', line=dict(color=color, width=2.5),
        name=f'{selected_word} ({result.upper()})'
    ))

    # Speech window
    fig.add_vrect(x0=VIDEO_START, x1=SPEECH_END,
                  fillcolor='rgba(70,130,180,0.07)', line_width=0,
                  annotation_text='Speech', annotation_position='top left')
    fig.add_vline(x=VIDEO_START, line_dash='dash', line_color='royalblue', line_width=1.5)
    fig.add_vline(x=SPEECH_END,  line_dash='dash', line_color='royalblue', line_width=1.5)

    # Word spoken timestamp
    if utc:
        fig.add_vline(x=utc, line_dash='dot', line_color=color, line_width=2.5,
                      annotation_text=f'"{selected_word}" spoken',
                      annotation_position='top right',
                      annotation_font_color=color)

    # 50¢ line
    fig.add_hline(y=50, line_dash='dot', line_color='gray', line_width=0.8, opacity=0.4)

    # Get price at word-spoken moment
    price_at_word = None
    if utc and not df.empty:
        before = df[df['ts'] <= utc]
        if not before.empty:
            price_at_word = before['mid'].dropna().iloc[-1] if not before['mid'].dropna().empty else None

    title_extra = ''
    if price_at_word is not None:
        mm = elapsed // 60
        ss = elapsed % 60
        direction = 'YES' if result == 'yes' else 'NO'
        title_extra = (f' | Price when spoken: {price_at_word:.0f}¢ → resolved {direction} (99¢)'
                       if result == 'yes'
                       else f' | Price when spoken: {price_at_word:.0f}¢ → resolved NO (1¢)')

    fig.update_layout(
        title=dict(
            text=f'{selected_word}{title_extra}',
            font=dict(size=13)
        ),
        xaxis=dict(
            title='Time UTC',
            tickformat='%H:%M',
            showgrid=True, gridcolor='rgba(80,80,80,0.3)'
        ),
        yaxis=dict(
            title='YES price (¢)',
            range=[-3, 105],
            showgrid=True, gridcolor='rgba(80,80,80,0.3)'
        ),
        plot_bgcolor='#0f0f1a',
        paper_bgcolor='#0f0f1a',
        font=dict(color='#e0e0e0'),
        legend=dict(orientation='h', y=1.02, x=0),
        margin=dict(l=50, r=20, t=60, b=40),
        height=340,
    )
    return fig

def build_volume_chart(selected_word):
    meta = mdict.get(selected_word)
    if not meta:
        return None
    df = load_candles(meta['ticker'])
    if df is None or df.empty:
        return None
    color = WORD_COLORS.get(selected_word, '#ff7f0e')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['ts'], y=df['vol'],
        marker_color=color, opacity=0.7, name='Volume'
    ))
    elapsed, utc = word_ts.get(selected_word, (None, None))
    if utc:
        fig.add_vline(x=utc, line_dash='dot', line_color=color, line_width=2)
    fig.add_vline(x=VIDEO_START, line_dash='dash', line_color='royalblue', line_width=1.2)
    fig.add_vline(x=SPEECH_END,  line_dash='dash', line_color='royalblue', line_width=1.2)
    fig.update_layout(
        xaxis=dict(tickformat='%H:%M', showgrid=True, gridcolor='rgba(80,80,80,0.3)'),
        yaxis=dict(title='Volume', showgrid=True, gridcolor='rgba(80,80,80,0.3)'),
        plot_bgcolor='#0f0f1a',
        paper_bgcolor='#0f0f1a',
        font=dict(color='#e0e0e0'),
        margin=dict(l=50, r=20, t=20, b=40),
        height=160,
        showlegend=False,
    )
    return fig

# ── Streamlit layout ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: #0a0a14; }
  section[data-testid="stSidebar"] { background: #0f0f1a; }
  .block-container { padding-top: 1rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 📊 Kalshi Transcript Viewer — Black History Month Feb 18 2026")
st.markdown(
    f"**Event:** `{EVENT}` &nbsp;|&nbsp; **Speech:** 20:12–20:59 UTC &nbsp;|&nbsp; "
    f"**Transcript:** {len(segments)} segments &nbsp;|&nbsp; "
    f"*Video time 00:00 = 20:12 UTC (calibrated from market data)*",
    unsafe_allow_html=True
)

# Word selector and info
col_sel, col_info = st.columns([2, 5])
with col_sel:
    selected = st.selectbox(
        "Select keyword:",
        list(FOCUS.keys()),
        format_func=lambda w: f"{'✅' if RESULT[w]=='yes' else '❌'} {w}"
    )

elapsed, utc = word_ts.get(selected, (None, None))
result = RESULT.get(selected, '?')

with col_info:
    if utc:
        mm, ss = elapsed // 60, elapsed % 60
        st.markdown(f"""
<div style="background:#1a1a2e;border-radius:6px;padding:10px 16px;margin-top:4px">
  <span style="color:{WORD_COLORS[selected]};font-size:18px;font-weight:bold">"{selected}"</span>
  &nbsp;&nbsp;
  <span style="color:#aaa">spoken at video</span>
  <span style="color:#fff;font-weight:bold"> {mm:02d}:{ss:02d}</span>
  <span style="color:#aaa"> = </span>
  <span style="color:#fff;font-weight:bold">{utc.strftime('%H:%M:%S')} UTC</span>
  &nbsp;&nbsp;
  <span style="background:{'#1a4a1a' if result=='yes' else '#4a1a1a'};
               color:{'#4caf50' if result=='yes' else '#f44336'};
               padding:3px 10px;border-radius:10px;font-weight:bold">
    Resolved {'YES ✅' if result=='yes' else 'NO ❌'}
  </span>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div style="background:#1a1a2e;border-radius:6px;padding:10px 16px;margin-top:4px">
  <span style="color:{WORD_COLORS[selected]};font-size:18px;font-weight:bold">"{selected}"</span>
  &nbsp;&nbsp;
  <span style="color:#888">Not spoken by Trump</span>
  &nbsp;&nbsp;
  <span style="background:#4a1a1a;color:#f44336;padding:3px 10px;border-radius:10px;font-weight:bold">
    Resolved NO ❌
  </span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Main layout: transcript left, charts right
left, right = st.columns([5, 4], gap="medium")

with left:
    st.markdown(f"### Transcript — {selected} highlighted")
    transcript_html = build_transcript_html(selected)
    st.components.v1.html(f"""
<div style="height:700px;overflow-y:auto;background:#0a0a14;
            border:1px solid #333;border-radius:6px;padding:8px">
  {transcript_html}
</div>
""", height=720)

with right:
    st.markdown("### Market price")
    price_fig = build_price_chart(selected)
    if price_fig:
        st.plotly_chart(price_fig, use_container_width=True)

    st.markdown("### Volume")
    vol_fig = build_volume_chart(selected)
    if vol_fig:
        st.plotly_chart(vol_fig, use_container_width=True)

    # Show pre-word excerpt
    st.markdown("### What was being said just before...")
    if elapsed:
        # collect last ~400 chars of Trump speech up to this point
        trump_text = []
        for seg in segments:
            if seg['elapsed'] <= elapsed and 'trump' in seg['speaker'].lower():
                trump_text.append(seg['text'].replace('\n', ' '))
        combined = ' '.join(trump_text)[-450:]
        color = WORD_COLORS[selected]
        terms = FOCUS[selected]
        for term in terms:
            pat = re.compile(re.escape(term), re.IGNORECASE)
            combined = pat.sub(
                f'**`{term.upper()}`**', combined
            )
        st.markdown(f"""
<div style="background:#1a1a2e;border-radius:6px;padding:12px 16px;
            border-left:4px solid {color};font-size:13px;
            color:#d0d0d0;line-height:1.7">
  <em>...{combined}</em>
</div>
""", unsafe_allow_html=True)
    else:
        st.info(f'"{selected}" was never said by Trump in this speech.')
