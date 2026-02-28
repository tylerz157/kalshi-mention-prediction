"""Generate a static HTML viewer: transcript timeline left, synced market chart right."""
import ast, json, os, re
import pandas as pd

EVENT       = 'KXTRUMPMENTION-26FEB19'
TRANSCRIPT  = 'transcripts/black_history_month_2026_02_19.txt'
VIDEO_START = pd.Timestamp('2026-02-18 20:12:00', tz='UTC')
SPEECH_END  = pd.Timestamp('2026-02-18 20:59:00', tz='UTC')

FOCUS = {
    'Democrat':             (['democrat'],        'yes', '#2ca02c'),
    'Election':             (['election'],        'yes', '#17becf'),
    'Hottest':              (['hottest'],         'yes', '#ff7f0e'),
    'Biden':                (['biden'],           'yes', '#9467bd'),
    'Stock Market':         (['stock market'],    'yes', '#1f77b4'),
    'ICE / National Guard': (['national guard'],  'yes', '#8c564b'),
    'Crime / Criminal':     (['criminal'],        'yes', '#e377c2'),
    'Bad Bunny':            (['bad bunny'],       'no',  '#d62728'),
    'Epstein':              (['epstein'],         'no',  '#aaaaaa'),
    'Supreme Court':        (['supreme court'],   'no',  '#f7b6d2'),
    'Crypto / Bitcoin':     (['crypto','bitcoin'],'no',  '#bcbd22'),
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
        segments.append({'elapsed': elapsed,
                         'utc': VIDEO_START + pd.Timedelta(seconds=elapsed),
                         'speaker': current_speaker, 'text': text})
segments.sort(key=lambda x: x['elapsed'])

# ── Find word timestamps ──────────────────────────────────────────────────────
def find_word_ts(terms):
    for seg in segments:
        if 'trump' not in seg['speaker'].lower(): continue
        tl = seg['text'].lower()
        for term in terms:
            if term in tl:
                return seg['elapsed'], seg['utc']
    return None, None

word_ts = {w: find_word_ts(t) for w, (t, _, __) in FOCUS.items()}

# ── Load market data ──────────────────────────────────────────────────────────
def pn(v):
    try: return ast.literal_eval(v)
    except: return None

with open('../past_kalshi_markets/trump_mention_markets.json') as f:
    markets = json.load(f)
mdict = {m['yes_sub_title']: m for m in markets if m['event_ticker'] == EVENT}

chart_data = {}
for word, (terms, result, color) in FOCUS.items():
    meta = mdict.get(word)
    if not meta: continue
    path = f'../past_kalshi_markets/{meta["ticker"]}_candlesticks.csv'
    if not os.path.exists(path): continue
    df = pd.read_csv(path)
    df['ts']  = pd.to_datetime(df['end_period_ts'].astype(int), unit='s', utc=True)
    df['ask'] = df['yes_ask'].apply(pn).apply(lambda d: d.get('close') if isinstance(d, dict) else None)
    df['bid'] = df['yes_bid'].apply(pn).apply(lambda d: d.get('close') if isinstance(d, dict) else None)
    df['pc']  = df['price'].apply(pn).apply(lambda d: d.get('close') if isinstance(d, dict) else None)
    df['mid'] = df.apply(lambda r: r['pc'] if r['pc'] else (
        (r['ask']+r['bid'])/2 if r['ask'] and r['bid'] else None), axis=1)
    df['vol'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    df = df[df['ts'].between(VIDEO_START - pd.Timedelta('20min'),
                              SPEECH_END  + pd.Timedelta('10min'))].sort_values('ts')
    chart_data[word] = {
        'ts':  [t.isoformat() for t in df['ts']],
        'mid': [round(v,1) if pd.notna(v) else None for v in df['mid']],
        'ask': [round(v,1) if pd.notna(v) else None for v in df['ask']],
        'bid': [round(v,1) if pd.notna(v) else None for v in df['bid']],
        'vol': [int(v) for v in df['vol']],
    }
print(f'Parsed {len(segments)} segments, {len(chart_data)} word charts')

# ── Build transcript HTML ─────────────────────────────────────────────────────
def esc(s): return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

rows = []
for seg in segments:
    is_trump = 'trump' in seg['speaker'].lower()
    mm, ss = seg['elapsed']//60, seg['elapsed']%60
    text = esc(seg['text'])
    for word, (terms, result, color) in FOCUS.items():
        for term in terms:
            safe = word.replace("'","\\'").replace('"','&quot;')
            text = re.compile(r'(?i)('+re.escape(term)+r')').sub(
                f'<mark class="kw" data-word="{safe}" style="background:{color};'
                f'color:#000;border-radius:2px;padding:0 3px;cursor:pointer" '
                f'onclick="selectWord(\'{safe}\')">'r'\1</mark>', text)
    cls = 'trump' if is_trump else 'other'
    rows.append(
        f'<div class="seg {cls}" data-elapsed="{seg["elapsed"]}" data-utc="{seg["utc"].isoformat()}">'
        f'<div class="tc"><span class="vt">{mm:02d}:{ss:02d}</span>'
        f'<span class="ut">{seg["utc"].strftime("%H:%M")}</span></div>'
        f'<div class="bc"><span class="sp">{esc(seg["speaker"])}</span>'
        f'<div class="tx">{text}</div></div></div>'
    )
transcript_html = '\n'.join(rows)

# ── Serialize ─────────────────────────────────────────────────────────────────
focus_js   = json.dumps({w:{'terms':t,'result':r,'color':c} for w,(t,r,c) in FOCUS.items()})
chart_js   = json.dumps(chart_data)
word_ts_js = json.dumps({w:{'elapsed':e,'utc':u.isoformat() if u else None}
                          for w,(e,u) in word_ts.items()})
t0 = VIDEO_START.isoformat()
t1 = SPEECH_END.isoformat()

# ── HTML ──────────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Kalshi Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ height: 100%; overflow: hidden; background: #090912; color: #ddd;
    font-family: system-ui, sans-serif; }}

  /* ── top bar ── */
  #topbar {{ height: 36px; display: flex; align-items: center; gap: 16px;
    padding: 0 14px; background: #0d0d1c; border-bottom: 1px solid #1e1e3a;
    flex-shrink: 0; }}
  #topbar h1 {{ font-size: 13px; color: #fff; white-space: nowrap; }}
  #topbar small {{ font-size: 11px; color: #555; }}

  /* ── tabs ── */
  #tabs {{ height: 34px; display: flex; align-items: center; gap: 5px;
    padding: 0 12px; background: #0b0b18; border-bottom: 1px solid #1e1e3a;
    flex-shrink: 0; overflow-x: auto; }}
  .tab {{ padding: 2px 10px; border-radius: 10px; font-size: 11px; cursor: pointer;
    border: 1px solid #333; color: #999; white-space: nowrap; flex-shrink: 0;
    transition: all .12s; }}
  .tab.active {{ font-weight: 700; color: #000 !important; border-color: transparent; }}

  /* ── main ── */
  #main {{ display: flex; height: calc(100vh - 70px); }}

  /* ── left: transcript ── */
  #transcript {{ width: 48%; overflow-y: auto; border-right: 1px solid #1a1a30; }}
  .seg {{ display: flex; border-left: 3px solid transparent; }}
  .seg.trump {{ background: #10102a; }}
  .seg.other {{ background: #0c0c1e; opacity: .4; }}
  .seg.cur    {{ background: #18183a !important; opacity: 1 !important; }}
  .tc {{ width: 60px; min-width: 60px; padding: 5px 5px 5px 7px;
    display: flex; flex-direction: column; align-items: flex-end;
    border-right: 1px solid #181830; gap: 1px; }}
  .vt {{ font-size: 10px; color: #556; font-family: monospace; font-weight: 600; }}
  .ut {{ font-size: 9px;  color: #334; font-family: monospace; }}
  .bc {{ flex: 1; padding: 5px 9px; }}
  .sp {{ font-size: 10px; font-weight: 700; display: block; margin-bottom: 1px; }}
  .trump .sp {{ color: #6080c0; }}
  .other .sp {{ color: #444; }}
  .tx {{ font-size: 12px; line-height: 1.6; color: #bbb; }}
  .trump .tx {{ color: #d5d5e5; }}

  /* ── right ── */
  #right {{ width: 52%; display: flex; flex-direction: column; }}

  /* now-bar */
  #nowbar {{ height: 42px; flex-shrink: 0; display: flex; align-items: center;
    gap: 14px; padding: 0 14px; background: #0d0d1c; border-bottom: 1px solid #1e1e3a; }}
  #now-ts {{ font-size: 22px; font-weight: 700; color: #fff; font-family: monospace; }}
  #now-utc {{ font-size: 11px; color: #555; font-family: monospace; }}
  #now-price {{ font-size: 18px; font-weight: 700; font-family: monospace;
    padding: 1px 10px; border-radius: 5px; background: #141428; }}
  #now-word {{ font-size: 12px; font-weight: 600; }}
  .badge {{ padding: 1px 8px; border-radius: 8px; font-size: 11px; font-weight: 700; margin-left:5px; }}
  .yes {{ background: #143014; color: #4caf50; }}
  .no  {{ background: #301414; color: #f44336; }}

  /* charts fill remaining space */
  #charts {{ flex: 1; min-height: 0; display: flex; flex-direction: column; }}
  #price-chart {{ flex: 1; min-height: 0; }}
  #vol-chart   {{ height: 22%; flex-shrink: 0; border-top: 1px solid #1a1a30; }}
</style>
</head>
<body>

<div id="topbar">
  <h1>📊 Kalshi — Black History Month · Feb 18 2026</h1>
  <small>{EVENT} · 20:12–20:59 UTC · scroll transcript → moves chart cursor · click chart → jumps transcript</small>
</div>
<div id="tabs"></div>

<div id="main">
  <div id="transcript">{transcript_html}</div>
  <div id="right">
    <div id="nowbar">
      <div>
        <div id="now-ts">00:00</div>
        <div id="now-utc">20:12:00 UTC</div>
      </div>
      <div id="now-price" style="color:#888">—¢</div>
      <div><span id="now-word"></span><span id="now-badge" class="badge">?</span></div>
    </div>
    <div id="charts">
      <div id="price-chart"></div>
      <div id="vol-chart"></div>
    </div>
  </div>
</div>

<script>
const FOCUS  = {focus_js};
const CD     = {chart_js};
const WTS    = {word_ts_js};
const T0     = "{t0}";
const T1     = "{t1}";
const T0ms   = new Date(T0).getTime();

let curWord = null, curElapsed = 0, chartsReady = false;

// ── tabs ───────────────────────────────────────────────────────────────────
const tabsEl = document.getElementById('tabs');
Object.entries(FOCUS).forEach(([w, info]) => {{
  const t = document.createElement('div');
  t.className = 'tab'; t.id = 'tab-'+w;
  t.textContent = (info.result==='yes'?'✅ ':'❌ ')+w;
  t.style.cssText = `background:${{info.color}}22;border-color:${{info.color}}55`;
  t.onclick = () => selectWord(w);
  tabsEl.appendChild(t);
}});

// ── helpers ────────────────────────────────────────────────────────────────
const elToUTC = e => new Date(T0ms + e*1000).toISOString();
const allSegs = Array.from(document.querySelectorAll('.seg'));

function priceAt(word, elapsed) {{
  const cd = CD[word]; if (!cd) return null;
  const tgt = T0ms + elapsed*1000;
  let best=null, bestD=Infinity;
  for (let i=0;i<cd.ts.length;i++) {{
    const d = Math.abs(new Date(cd.ts[i]).getTime()-tgt);
    if (d<bestD && cd.mid[i]!=null) {{ bestD=d; best=cd.mid[i]; }}
  }}
  return best;
}}

// ── now-bar ────────────────────────────────────────────────────────────────
function updateNowBar(elapsed) {{
  const mm=String(Math.floor(elapsed/60)).padStart(2,'0');
  const ss=String(Math.floor(elapsed%60)).padStart(2,'0');
  document.getElementById('now-ts').textContent = mm+':'+ss;
  document.getElementById('now-utc').textContent =
    new Date(T0ms+elapsed*1000).toISOString().slice(11,19)+' UTC';
  if (curWord) {{
    const p = priceAt(curWord, elapsed);
    const el = document.getElementById('now-price');
    el.textContent = p!=null ? p.toFixed(0)+'¢' : '—¢';
    el.style.color = FOCUS[curWord].color;
  }}
}}

// ── chart cursor ───────────────────────────────────────────────────────────
function moveCursor(elapsed) {{
  if (!chartsReady) return;
  const utc = elToUTC(elapsed);
  Plotly.relayout('price-chart', {{'shapes[0].x0':utc,'shapes[0].x1':utc}});
  Plotly.relayout('vol-chart',   {{'shapes[0].x0':utc,'shapes[0].x1':utc}});
}}

// ── active segment highlight ───────────────────────────────────────────────
let prevSeg = null;
function setActive(seg) {{
  if (prevSeg===seg) return;
  if (prevSeg) prevSeg.classList.remove('cur');
  prevSeg = seg;
  if (seg) {{ seg.classList.add('cur');
    seg.style.borderLeftColor = curWord ? FOCUS[curWord].color+'99' : '#555'; }}
}}

// ── transcript scroll → cursor ─────────────────────────────────────────────
const txEl = document.getElementById('transcript');
let raf = false;
txEl.addEventListener('scroll', () => {{
  if (raf) return; raf = true;
  requestAnimationFrame(() => {{
    raf = false;
    const top = txEl.getBoundingClientRect().top;
    let found = null;
    for (const s of allSegs) {{
      if (s.getBoundingClientRect().bottom > top+4) {{ found=s; break; }}
    }}
    if (!found) return;
    const e = parseInt(found.dataset.elapsed);
    if (e===curElapsed) return;
    curElapsed = e;
    updateNowBar(e);
    moveCursor(e);
    setActive(found);
  }});
}});

// ── selectWord ─────────────────────────────────────────────────────────────
function selectWord(word) {{
  if (!FOCUS[word]) return;
  curWord = word;
  const info=FOCUS[word], color=info.color, wts=WTS[word];

  document.querySelectorAll('.tab').forEach(t=>{{ t.classList.remove('active'); t.style.color=''; }});
  const tab=document.getElementById('tab-'+word);
  if (tab) {{ tab.classList.add('active'); tab.style.background=color; tab.style.color='#000'; }}

  document.querySelectorAll('.seg.trump').forEach(s=>s.style.borderLeftColor=color+'55');

  document.getElementById('now-word').textContent=word;
  document.getElementById('now-word').style.color=color;
  const badge=document.getElementById('now-badge');
  badge.textContent=info.result.toUpperCase();
  badge.className='badge '+info.result;

  drawCharts(word);
  updateNowBar(curElapsed);

  // scroll to first mark
  const mark = document.querySelector(`mark.kw[data-word="${{word}}"]`);
  if (mark) mark.closest('.seg').scrollIntoView({{behavior:'smooth',block:'center'}});
}}

// ── draw charts ────────────────────────────────────────────────────────────
function drawCharts(word) {{
  const info=FOCUS[word], color=info.color, wts=WTS[word], cd=CD[word];
  if (!cd) return;
  const bg='#090912', grid='rgba(55,55,80,0.4)', cursor=elToUTC(curElapsed);

  // shapes: [0]=cursor, [1]=fill, [2]=start, [3]=end, [4]=spoken
  const shapes = [
    {{type:'line',xref:'x',yref:'paper',x0:cursor,x1:cursor,y0:0,y1:1,
      line:{{color:'rgba(255,255,255,0.6)',width:1.5}}}},
    {{type:'rect',xref:'x',yref:'paper',x0:T0,x1:T1,y0:0,y1:1,
      fillcolor:'rgba(40,70,160,0.06)',line:{{width:0}}}},
    {{type:'line',xref:'x',yref:'paper',x0:T0,x1:T0,y0:0,y1:1,
      line:{{color:'#2a3a6a',width:1,dash:'dash'}}}},
    {{type:'line',xref:'x',yref:'paper',x0:T1,x1:T1,y0:0,y1:1,
      line:{{color:'#2a3a6a',width:1,dash:'dash'}}}},
  ];
  const anns = [];
  if (wts && wts.utc) {{
    shapes.push({{type:'line',xref:'x',yref:'paper',x0:wts.utc,x1:wts.utc,y0:0,y1:1,
      line:{{color:color,width:2,dash:'dot'}}}});
    anns.push({{x:wts.utc,y:0.97,xref:'x',yref:'paper',text:'spoken',
      showarrow:false,yanchor:'top',font:{{color:color,size:9}}}});
  }}

  const base = {{
    paper_bgcolor:bg, plot_bgcolor:bg, font:{{color:'#999',size:10}},
    margin:{{l:40,r:8,t:30,b:30}}, showlegend:false,
    xaxis:{{tickformat:'%H:%M',gridcolor:grid,showgrid:true,zeroline:false,color:'#555'}},
    shapes, autosize:true,
  }};

  Plotly.react('price-chart', [
    {{x:[...cd.ts,...[...cd.ts].reverse()], y:[...cd.ask,...[...cd.bid].reverse()],
      fill:'toself',fillcolor:color+'12',line:{{color:'transparent'}},hoverinfo:'skip'}},
    {{x:cd.ts,y:cd.mid,mode:'lines',line:{{color,width:2.5}},
      hovertemplate:'%{{x|%H:%M:%S}}<br><b>%{{y:.0f}}¢</b><extra></extra>'}},
  ], {{...base,
    title:{{text:`${{word}} — Resolved ${{info.result.toUpperCase()}}`,font:{{color:'#ccc',size:12}}}},
    yaxis:{{range:[-2,106],title:'YES ¢',gridcolor:grid,zeroline:false,color:'#555'}},
    annotations:anns,
  }}, {{responsive:true}});

  Plotly.react('vol-chart', [
    {{x:cd.ts,y:cd.vol,type:'bar',marker:{{color,opacity:0.55}},
      hovertemplate:'%{{x|%H:%M}}: %{{y}}<extra></extra>'}},
  ], {{...base,
    margin:{{l:40,r:8,t:8,b:30}},
    yaxis:{{title:'',gridcolor:grid,zeroline:false,color:'#555'}},
    shapes:shapes.slice(0,4),
  }}, {{responsive:true}});

  chartsReady = true;

  // click chart → scroll transcript
  document.getElementById('price-chart').on('plotly_click', data => {{
    if (!data.points.length) return;
    const ms = new Date(data.points[0].x).getTime();
    const e  = (ms - T0ms)/1000;
    let best=allSegs[0], bd=Infinity;
    allSegs.forEach(s => {{
      const d=Math.abs(parseInt(s.dataset.elapsed)-e);
      if (d<bd) {{ bd=d; best=s; }}
    }});
    best.scrollIntoView({{behavior:'smooth',block:'center'}});
  }});
}}

// ── init ───────────────────────────────────────────────────────────────────
selectWord('Democrat');
txEl.dispatchEvent(new Event('scroll'));
</script>
</body>
</html>
"""

with open('viewer.html', 'w', encoding='utf-8') as f:
    f.write(html)
print(f'Saved viewer.html ({len(html)//1024} KB)')
