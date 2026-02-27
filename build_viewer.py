"""Generate a static HTML viewer: transcript timeline left, synced market chart right."""
import ast, json, os, re
import pandas as pd

EVENT       = 'KXTRUMPMENTION-26FEB19'
TRANSCRIPT  = 'transcripts/timestamped/black_history_month_2026_02_19.txt'
VIDEO_START = pd.Timestamp('2026-02-18 20:12:00', tz='UTC')
SPEECH_END  = pd.Timestamp('2026-02-18 20:59:00', tz='UTC')

FOCUS = {
    'Democrat':             (['democrat'],       'yes', '#2ca02c'),
    'Election':             (['election'],       'yes', '#17becf'),
    'Hottest':              (['hottest'],        'yes', '#ff7f0e'),
    'Biden':                (['biden'],          'yes', '#9467bd'),
    'Stock Market':         (['stock market'],   'yes', '#1f77b4'),
    'ICE / National Guard': (['national guard'], 'yes', '#8c564b'),
    'Crime / Criminal':     (['criminal'],       'yes', '#e377c2'),
    'Bad Bunny':            (['bad bunny'],      'no',  '#d62728'),
    'Epstein':              (['epstein'],        'no',  '#aaaaaa'),
    'Supreme Court':        (['supreme court'],  'no',  '#f7b6d2'),
    'Crypto / Bitcoin':     (['crypto','bitcoin'],'no', '#bcbd22'),
}

# ── Parse transcript ──────────────────────────────────────────────────────────
with open(TRANSCRIPT, encoding='utf-8') as f:
    raw = f.read()
body = raw.split('=' * 10)[-1].strip()
token_re = re.compile(
    r'(?:^([A-Za-z][A-Za-z .0-9]*?)\s*)?\((\d{1,2}):(\d{2})\)\s*:?',
    re.MULTILINE
)
tokens_found = list(token_re.finditer(body))
segments = []
current_speaker = 'Unknown'
for i, tok in enumerate(tokens_found):
    if tok.group(1):
        current_speaker = tok.group(1).strip()
    elapsed = int(tok.group(2)) * 60 + int(tok.group(3))
    start = tok.end()
    end   = tokens_found[i+1].start() if i+1 < len(tokens_found) else len(body)
    text  = body[start:end].strip()
    if text:
        segments.append({
            'elapsed': elapsed,
            'utc': VIDEO_START + pd.Timedelta(seconds=elapsed),
            'speaker': current_speaker,
            'text': text,
        })
segments.sort(key=lambda x: x['elapsed'])
print(f'Parsed {len(segments)} segments')

# ── Find word timestamps ──────────────────────────────────────────────────────
def find_word_ts(terms):
    for seg in segments:
        if 'trump' not in seg['speaker'].lower():
            continue
        tl = seg['text'].lower()
        for term in terms:
            if term in tl:
                return seg['elapsed'], seg['utc']
    return None, None

word_ts = {}
for word, (terms, result, color) in FOCUS.items():
    elapsed, utc = find_word_ts(terms)
    word_ts[word] = (elapsed, utc)

# ── Load market data ──────────────────────────────────────────────────────────
def pn(v):
    try: return ast.literal_eval(v)
    except: return None

with open('data/kalshi/trump_mention_markets.json') as f:
    markets = json.load(f)
mdict = {m['yes_sub_title']: m for m in markets if m['event_ticker'] == EVENT}

chart_data = {}
for word in FOCUS:
    meta = mdict.get(word)
    if not meta:
        continue
    path = f'data/kalshi/{meta["ticker"]}_candlesticks.csv'
    if not os.path.exists(path):
        continue
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
        VIDEO_START - pd.Timedelta('20min'),
        SPEECH_END  + pd.Timedelta('10min')
    )
    df = df[window].sort_values('ts')
    chart_data[word] = {
        'ts':  [t.isoformat() for t in df['ts']],
        'mid': [round(v, 1) if pd.notna(v) else None for v in df['mid']],
        'ask': [round(v, 1) if pd.notna(v) else None for v in df['ask']],
        'bid': [round(v, 1) if pd.notna(v) else None for v in df['bid']],
        'vol': [int(v) for v in df['vol']],
    }
print(f'Loaded chart data for {len(chart_data)} words')

# ── Build transcript HTML ─────────────────────────────────────────────────────
def escape(s):
    return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

transcript_rows = []
for seg in segments:
    is_trump = 'trump' in seg['speaker'].lower()
    mm, ss   = seg['elapsed'] // 60, seg['elapsed'] % 60
    utc_iso  = seg['utc'].isoformat()
    utc_str  = seg['utc'].strftime('%H:%M')
    text_esc = escape(seg['text'])
    for word, (terms, result, color) in FOCUS.items():
        for term in terms:
            pat = re.compile(r'(?i)(' + re.escape(term) + r')')
            safe_word = word.replace("'","\\'").replace('"','&quot;')
            text_esc = pat.sub(
                f'<mark class="kw" data-word="{safe_word}" '
                f'style="background:{color};color:#000;border-radius:2px;'
                f'padding:0 3px;cursor:pointer" '
                f'onclick="selectWord(\'{safe_word}\')">'
                r'\1</mark>',
                text_esc
            )
    cls = 'trump' if is_trump else 'other'
    sp  = escape(seg['speaker'])
    transcript_rows.append(
        f'<div class="seg {cls}" data-elapsed="{seg["elapsed"]}" data-utc="{utc_iso}">'
        f'<div class="time-col">'
        f'  <span class="vidts">{mm:02d}:{ss:02d}</span>'
        f'  <span class="utcts">{utc_str}</span>'
        f'</div>'
        f'<div class="text-col">'
        f'  <span class="sp">{sp}</span>'
        f'  <div class="tx">{text_esc}</div>'
        f'</div>'
        f'</div>'
    )
transcript_html = '\n'.join(transcript_rows)

# ── Serialize for JS ──────────────────────────────────────────────────────────
focus_js        = json.dumps({w: {'terms':t,'result':r,'color':c} for w,(t,r,c) in FOCUS.items()})
chart_js        = json.dumps(chart_data)
word_ts_js      = json.dumps({w: {'elapsed':e,'utc':u.isoformat() if u else None} for w,(e,u) in word_ts.items()})
speech_start_js = VIDEO_START.isoformat()
speech_end_js   = SPEECH_END.isoformat()

# ── HTML ──────────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Kalshi Viewer — Black History Month 2026</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#090912;color:#ddd;font-family:system-ui,sans-serif;overflow:hidden}}

/* ── header ── */
#header{{padding:8px 16px;background:#0d0d1c;border-bottom:1px solid #1e1e3a;
  display:flex;align-items:center;gap:20px;flex-wrap:wrap}}
#header h1{{font-size:14px;color:#fff;font-weight:600}}
#header span{{font-size:11px;color:#666}}

/* ── tabs ── */
#tabs{{display:flex;gap:5px;flex-wrap:wrap;padding:6px 16px;
  background:#0b0b18;border-bottom:1px solid #1e1e3a}}
.tab{{padding:3px 11px;border-radius:10px;font-size:11.5px;cursor:pointer;
  border:1px solid #333;color:#999;background:#141428;transition:all .12s;white-space:nowrap}}
.tab:hover{{filter:brightness(1.4)}}
.tab.active{{color:#000!important;font-weight:700;border-color:transparent}}

/* ── layout ── */
#main{{display:flex;height:calc(100vh - 68px)}}

/* ── transcript (left) ── */
#transcript{{width:50%;overflow-y:auto;position:relative;border-right:1px solid #1a1a30}}
.seg{{display:flex;gap:0;border-left:3px solid transparent;
  transition:background .15s,opacity .15s;cursor:default}}
.seg.trump{{background:#10102a;border-left-color:#252548}}
.seg.other{{background:#0c0c1e;opacity:0.45}}
.seg.active-seg{{background:#18183a!important;border-left-color:var(--active-color,#888)!important;opacity:1!important}}
.time-col{{width:72px;min-width:72px;padding:7px 6px 7px 8px;
  display:flex;flex-direction:column;align-items:flex-end;gap:1px;
  border-right:1px solid #1a1a2e}}
.vidts{{font-size:10px;color:#555;font-family:monospace;font-weight:600}}
.utcts{{font-size:9px;color:#333;font-family:monospace}}
.text-col{{flex:1;padding:6px 10px 6px 10px}}
.sp{{font-size:10px;font-weight:700;display:block;margin-bottom:2px}}
.trump .sp{{color:#7090d0}}
.other .sp{{color:#555}}
.tx{{font-size:12.5px;line-height:1.65;color:#bbb}}
.trump .tx{{color:#d8d8e8}}

/* ── right panel ── */
#right{{width:50%;display:flex;flex-direction:column}}

/* current-time bar */
#now-bar{{padding:7px 14px;background:#0d0d1c;border-bottom:1px solid #1e1e3a;
  display:flex;align-items:center;gap:14px;min-height:44px;flex-wrap:wrap}}
#now-time{{font-size:20px;font-weight:700;color:#fff;font-family:monospace}}
#now-utc{{font-size:12px;color:#666;font-family:monospace}}
#now-price{{font-size:18px;font-weight:700;padding:2px 12px;border-radius:6px;
  background:#1a1a30;font-family:monospace}}
#word-label{{font-size:12px;font-weight:600}}
.badge{{display:inline-block;padding:2px 9px;border-radius:8px;font-size:11px;font-weight:700;margin-left:6px}}
.yes{{background:#143014;color:#4caf50}}
.no {{background:#301414;color:#f44336}}

/* charts */
#charts{{flex:1;display:flex;flex-direction:column;overflow:hidden}}
#price-chart{{flex:1;min-height:0}}
#vol-chart{{height:120px;border-top:1px solid #1a1a30}}
</style>
</head>
<body>

<div id="header">
  <h1>📊 Kalshi — Black History Month  Feb 18 2026</h1>
  <span>{EVENT}  |  20:12–20:59 UTC  |  Scroll transcript to move chart cursor  |  Click chart to jump transcript</span>
</div>
<div id="tabs"></div>

<div id="main">

  <!-- LEFT: transcript timeline -->
  <div id="transcript">
    {transcript_html}
  </div>

  <!-- RIGHT: charts -->
  <div id="right">
    <div id="now-bar">
      <div>
        <div id="now-time">00:00</div>
        <div id="now-utc">20:12:00 UTC</div>
      </div>
      <div id="now-price">—¢</div>
      <div>
        <span id="word-label">Democrat</span>
        <span id="word-badge" class="badge yes">YES</span>
      </div>
      <div style="font-size:11px;color:#444;margin-left:auto">
        ← scroll transcript to seek<br>click chart to jump
      </div>
    </div>
    <div id="charts">
      <div id="price-chart"></div>
      <div id="vol-chart"></div>
    </div>
  </div>

</div>

<script>
const FOCUS        = {focus_js};
const CHART_DATA   = {chart_js};
const WORD_TS      = {word_ts_js};
const SPEECH_START = "{speech_start_js}";
const SPEECH_END   = "{speech_end_js}";
const VIDEO_START_MS = new Date(SPEECH_START).getTime();

// ── Build tabs ─────────────────────────────────────────────────────────────
const tabsEl = document.getElementById('tabs');
Object.entries(FOCUS).forEach(([word, info]) => {{
  const t = document.createElement('div');
  t.className = 'tab';
  t.id = 'tab-' + word;
  t.textContent = (info.result==='yes'?'✅ ':'❌ ') + word;
  t.style.background = info.color + '20';
  t.style.borderColor = info.color + '55';
  t.addEventListener('click', () => selectWord(word));
  tabsEl.appendChild(t);
}});

// ── State ──────────────────────────────────────────────────────────────────
let currentWord    = null;
let currentElapsed = 0;
let chartsReady    = false;

// ── Cursor shape index: 0=cursor, 1=fill, 2=start, 3=end, 4=word-spoken
const CURSOR_IDX       = 0;
const SPOKEN_SHAPE_IDX = 4;

function elapsedToUTC(elapsed) {{
  return new Date(VIDEO_START_MS + elapsed * 1000);
}}
function utcToElapsed(utcDate) {{
  return (utcDate.getTime() - VIDEO_START_MS) / 1000;
}}

// Find mid price for a word at given elapsed seconds
function priceAtElapsed(word, elapsed) {{
  const cd = CHART_DATA[word];
  if (!cd) return null;
  const targetMs = VIDEO_START_MS + elapsed * 1000;
  let best = null, bestDiff = Infinity;
  for (let i = 0; i < cd.ts.length; i++) {{
    const diff = Math.abs(new Date(cd.ts[i]).getTime() - targetMs);
    if (diff < bestDiff && cd.mid[i] != null) {{
      bestDiff = diff;
      best = cd.mid[i];
    }}
  }}
  return best;
}}

// ── Update now-bar ─────────────────────────────────────────────────────────
function updateNowBar(elapsed) {{
  const mm = String(Math.floor(elapsed/60)).padStart(2,'0');
  const ss = String(Math.floor(elapsed%60)).padStart(2,'0');
  document.getElementById('now-time').textContent = mm+':'+ss;
  const utc = elapsedToUTC(elapsed);
  document.getElementById('now-utc').textContent =
    utc.toISOString().slice(11,19) + ' UTC';
  if (currentWord) {{
    const price = priceAtElapsed(currentWord, elapsed);
    const priceEl = document.getElementById('now-price');
    priceEl.textContent = price != null ? price.toFixed(0)+'¢' : '—¢';
    priceEl.style.color = FOCUS[currentWord].color;
  }}
}}

// ── Update chart cursor (shapes[0]) ───────────────────────────────────────
function updateCursor(elapsed) {{
  if (!chartsReady) return;
  const utcStr = elapsedToUTC(elapsed).toISOString();
  Plotly.relayout('price-chart', {{
    [`shapes[${{CURSOR_IDX}}].x0`]: utcStr,
    [`shapes[${{CURSOR_IDX}}].x1`]: utcStr,
  }});
  Plotly.relayout('vol-chart', {{
    [`shapes[${{CURSOR_IDX}}].x0`]: utcStr,
    [`shapes[${{CURSOR_IDX}}].x1`]: utcStr,
  }});
}}

// ── Highlight active transcript segment ───────────────────────────────────
let activeSeg = null;
function setActiveSeg(segEl) {{
  if (activeSeg === segEl) return;
  if (activeSeg) activeSeg.classList.remove('active-seg');
  activeSeg = segEl;
  if (segEl) {{
    segEl.classList.add('active-seg');
    if (currentWord) {{
      segEl.style.setProperty('--active-color', FOCUS[currentWord].color);
    }}
  }}
}}

// ── Transcript scroll → cursor ────────────────────────────────────────────
const transcriptEl = document.getElementById('transcript');
const allSegs = Array.from(document.querySelectorAll('.seg'));

let rafPending = false;
transcriptEl.addEventListener('scroll', () => {{
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => {{
    rafPending = false;
    const containerTop = transcriptEl.getBoundingClientRect().top;
    // Find first segment whose bottom edge is below the container top
    let found = null;
    for (const seg of allSegs) {{
      const r = seg.getBoundingClientRect();
      if (r.bottom > containerTop + 8) {{ found = seg; break; }}
    }}
    if (!found) return;
    const elapsed = parseInt(found.dataset.elapsed);
    if (elapsed === currentElapsed) return;
    currentElapsed = elapsed;
    updateNowBar(elapsed);
    updateCursor(elapsed);
    setActiveSeg(found);
  }});
}});

// ── Select word (tab click) ────────────────────────────────────────────────
function selectWord(word) {{
  if (!FOCUS[word]) return;
  currentWord = word;
  const info   = FOCUS[word];
  const color  = info.color;
  const tsInfo = WORD_TS[word];

  // Tabs
  document.querySelectorAll('.tab').forEach(t => {{
    t.classList.remove('active');
    t.style.color = '';
  }});
  const tab = document.getElementById('tab-' + word);
  if (tab) {{ tab.classList.add('active'); tab.style.background = color; tab.style.color='#000'; }}

  // Trump segments: update left border color
  document.querySelectorAll('.seg.trump').forEach(s => {{
    s.style.borderLeftColor = color + '60';
  }});

  // Update now-bar labels
  document.getElementById('word-label').textContent = word;
  document.getElementById('word-label').style.color = color;
  const badge = document.getElementById('word-badge');
  badge.textContent = info.result.toUpperCase();
  badge.className = 'badge ' + info.result;

  // Rebuild charts
  drawCharts(word);

  // Scroll to first keyword mark
  const mark = document.querySelector(`mark.kw[data-word="${{word}}"]`);
  if (mark) {{
    const segEl = mark.closest('.seg');
    if (segEl) {{
      segEl.scrollIntoView({{behavior:'smooth', block:'center'}});
    }}
  }}

  // Refresh now-bar with new word
  updateNowBar(currentElapsed);
}}

// ── Draw charts ────────────────────────────────────────────────────────────
function drawCharts(word) {{
  const info   = FOCUS[word];
  const color  = info.color;
  const tsInfo = WORD_TS[word];
  const cd     = CHART_DATA[word];
  if (!cd) return;

  const bg      = '#090912';
  const gridClr = 'rgba(60,60,90,0.4)';
  const cursorUtc = elapsedToUTC(currentElapsed).toISOString();

  // shapes[0] = cursor (white moving line)
  // shapes[1] = speech fill
  // shapes[2] = speech start
  // shapes[3] = speech end
  // shapes[4] = word-spoken dotted (if applicable)
  const staticShapes = [
    {{ type:'line', xref:'x', yref:'paper', x0:cursorUtc, x1:cursorUtc,
       y0:0, y1:1, line:{{color:'rgba(255,255,255,0.7)',width:1.5}} }},
    {{ type:'rect', xref:'x', yref:'paper', x0:SPEECH_START, x1:SPEECH_END,
       y0:0, y1:1, fillcolor:'rgba(50,80,180,0.07)', line:{{width:0}} }},
    {{ type:'line', xref:'x', yref:'paper', x0:SPEECH_START, x1:SPEECH_START,
       y0:0, y1:1, line:{{color:'#334488',width:1,dash:'dash'}} }},
    {{ type:'line', xref:'x', yref:'paper', x0:SPEECH_END, x1:SPEECH_END,
       y0:0, y1:1, line:{{color:'#334488',width:1,dash:'dash'}} }},
  ];
  const annotations = [];
  if (tsInfo && tsInfo.utc) {{
    staticShapes.push({{
      type:'line', xref:'x', yref:'paper', x0:tsInfo.utc, x1:tsInfo.utc,
      y0:0, y1:1, line:{{color:color, width:2, dash:'dot'}}
    }});
    annotations.push({{
      x:tsInfo.utc, y:0.98, xref:'x', yref:'paper',
      text:'spoken', showarrow:false, yanchor:'top',
      font:{{color:color, size:9}}, bgcolor:'#090912aa', bordercolor:color
    }});
  }}

  const baseLayout = {{
    paper_bgcolor:bg, plot_bgcolor:bg,
    font:{{color:'#aaa', size:10}},
    margin:{{l:42,r:10,t:28,b:28}},
    xaxis:{{tickformat:'%H:%M', gridcolor:gridClr, showgrid:true,
            zeroline:false, color:'#555'}},
    showlegend:false,
    shapes: staticShapes,
    annotations,
  }};

  // price traces
  const askFwd = cd.ask, bidRev=[...cd.bid].reverse();
  const tsFwd  = cd.ts,  tsRev =[...cd.ts].reverse();
  Plotly.react('price-chart', [
    {{ x:[...tsFwd,...tsRev], y:[...askFwd,...bidRev],
       fill:'toself', fillcolor:color+'15', line:{{color:'transparent'}},
       hoverinfo:'skip' }},
    {{ x:cd.ts, y:cd.mid, mode:'lines', line:{{color:color, width:2.5}},
       hovertemplate:'%{{x|%H:%M:%S}}<br><b>%{{y:.0f}}¢</b><extra></extra>' }},
  ], {{
    ...baseLayout,
    title:{{text:`${{word}} — Resolved ${{info.result.toUpperCase()}}`,
            font:{{color:'#ccc',size:12}}}},
    yaxis:{{range:[-2,106], title:'YES price (¢)', gridcolor:gridClr,
            zeroline:false, color:'#555', ticksuffix:'¢'}},
  }});

  // volume trace
  Plotly.react('vol-chart', [
    {{ x:cd.ts, y:cd.vol, type:'bar',
       marker:{{color:color, opacity:0.6}},
       hovertemplate:'%{{x|%H:%M}}: %{{y}}<extra></extra>' }},
  ], {{
    ...baseLayout,
    title:{{text:'Volume', font:{{color:'#666',size:10}}}},
    yaxis:{{title:'', gridcolor:gridClr, zeroline:false, color:'#555'}},
    shapes: staticShapes.filter(s=>s.type==='line'),
  }});

  chartsReady = true;

  // Bind chart click → scroll transcript
  const priceDiv = document.getElementById('price-chart');
  priceDiv.removeAllListeners && priceDiv.removeAllListeners('plotly_click');
  priceDiv.on('plotly_click', data => {{
    if (!data.points.length) return;
    const clickedMs = new Date(data.points[0].x).getTime();
    const clickedElapsed = (clickedMs - new Date(SPEECH_START).getTime()) / 1000;
    // Find nearest segment
    let best = allSegs[0], bestDiff = Infinity;
    allSegs.forEach(s => {{
      const diff = Math.abs(parseInt(s.dataset.elapsed) - clickedElapsed);
      if (diff < bestDiff) {{ bestDiff=diff; best=s; }}
    }});
    best.scrollIntoView({{behavior:'smooth', block:'center'}});
  }});
}}

// ── Init ───────────────────────────────────────────────────────────────────
selectWord('Democrat');
// Trigger one scroll read so cursor starts at top
transcriptEl.dispatchEvent(new Event('scroll'));
</script>
</body>
</html>
"""

out = 'viewer.html'
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'Saved: {out}  ({len(html)//1024} KB)')
