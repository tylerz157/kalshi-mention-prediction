"""
Explore a Kalshi mention-market event: price timelines, volume, and viability analysis.

Usage:
    python explore_event.py                          # list available events
    python explore_event.py KXTRUMPMENTION-26FEB28   # analyse SOTU
    python explore_event.py KXTRUMPMENTION-26FEB28 --transcript transcripts/senate_dems/some-slug.txt
"""

import argparse
import ast
import io
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


DATA_DIR = "../past_kalshi_markets"
TRANSCRIPT_DIR = "transcripts/senate_dems"
MARKETS_FILE = os.path.join(DATA_DIR, "trump_mention_markets.json")


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_nested(val):
    """ast.literal_eval on stringified dicts; return None on failure."""
    try:
        return ast.literal_eval(val)
    except Exception:
        return None


def load_candles(ticker):
    """Load a candlestick CSV into a DataFrame with parsed columns."""
    path = os.path.join(DATA_DIR, f"{ticker}_candlesticks.csv")
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["end_period_ts"].astype(int), unit="s", utc=True)

    for col in ["price", "yes_ask", "yes_bid"]:
        parsed = df[col].apply(parse_nested)
        df[f"{col}_close"] = parsed.apply(lambda d: d.get("close") if isinstance(d, dict) else None)

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

    # Mid price = average of ask and bid close (best available)
    df["mid"] = df.apply(
        lambda r: r["price_close"]
        if r["price_close"] is not None
        else (
            (r["yes_ask_close"] + r["yes_bid_close"]) / 2
            if r["yes_ask_close"] is not None and r["yes_bid_close"] is not None
            else r["yes_ask_close"] or r["yes_bid_close"]
        ),
        axis=1,
    )
    df["spread"] = df.apply(
        lambda r: r["yes_ask_close"] - r["yes_bid_close"]
        if r["yes_ask_close"] is not None and r["yes_bid_close"] is not None
        else None,
        axis=1,
    )

    return df.sort_values("ts").reset_index(drop=True)


def iso_to_dt(s):
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# ── listing ───────────────────────────────────────────────────────────────────

def list_events():
    with open(MARKETS_FILE) as f:
        markets = json.load(f)

    events = defaultdict(list)
    for m in markets:
        events[m["event_ticker"]].append(m)

    print(f"\n{'EVENT TICKER':<35} {'FILES':>6} {'YES':>5} {'NO':>5}  TITLE")
    print("-" * 100)
    for et, mlist in sorted(events.items()):
        saved = sum(1 for m in mlist if os.path.exists(os.path.join(DATA_DIR, f"{m['ticker']}_candlesticks.csv")))
        yes_n = sum(1 for m in mlist if m.get("result") == "yes")
        no_n  = sum(1 for m in mlist if m.get("result") == "no")
        title = mlist[0].get("title", "")[:55]
        marker = " *" if saved == len(mlist) else ""
        print(f"  {et:<33} {saved:>5}/{len(mlist):<3} {yes_n:>5} {no_n:>5}  {title}{marker}")

    print("\n* = all candlestick files present")
    print("\nAvailable transcripts (senate_dems):")
    slugs = sorted(f.replace(".txt","") for f in os.listdir(TRANSCRIPT_DIR) if f.endswith(".txt"))
    for s in slugs[:30]:
        print(f"  {s}")
    if len(slugs) > 30:
        print(f"  ... and {len(slugs)-30} more")


# ── main analysis ─────────────────────────────────────────────────────────────

def analyse(event_ticker, transcript_path=None):
    with open(MARKETS_FILE) as f:
        all_markets = json.load(f)

    markets = [m for m in all_markets if m["event_ticker"] == event_ticker]
    if not markets:
        print(f"No markets found for {event_ticker}")
        return

    title = markets[0].get("title", event_ticker)
    print(f"\n{'='*70}")
    print(f"EVENT: {event_ticker}")
    print(f"TITLE: {title}")
    print(f"Markets: {len(markets)}  YES: {sum(m['result']=='yes' for m in markets)}  NO: {sum(m['result']=='no' for m in markets)}")
    print(f"{'='*70}\n")

    # Load all candlesticks
    market_data = {}  # ticker -> (market_meta, df)
    for m in markets:
        df = load_candles(m["ticker"])
        if df is not None and len(df) > 0:
            market_data[m["ticker"]] = (m, df)

    if not market_data:
        print("No candlestick files found yet. Run fetch_kalshi_markets.py first.")
        return

    print(f"Loaded candlestick data for {len(market_data)}/{len(markets)} markets\n")

    # Determine speech window from data
    all_ts = pd.concat([df["ts"] for _, df in market_data.values()])
    speech_start = all_ts.min()
    speech_end   = all_ts.max()

    yes_markets = [(m, df) for t, (m, df) in market_data.items() if m.get("result") == "yes"]
    no_markets  = [(m, df) for t, (m, df) in market_data.items() if m.get("result") == "no"]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3,
                          height_ratios=[3, 1, 1])
    ax_price  = fig.add_subplot(gs[0, :])    # top full-width: price lines
    ax_vol    = fig.add_subplot(gs[1, :])    # middle: volume
    ax_react  = fig.add_subplot(gs[2, 0])   # bottom-left: reaction time
    ax_spread = fig.add_subplot(gs[2, 1])   # bottom-right: bid-ask spread

    # ── Panel 1: Price timelines ───────────────────────────────────────────
    for m, df in sorted(yes_markets, key=lambda x: x[0].get("volume", 0), reverse=True):
        word = m.get("yes_sub_title", m["ticker"])
        close_dt = iso_to_dt(m.get("close_time"))
        ax_price.plot(df["ts"], df["mid"], color="green", alpha=0.6, linewidth=1.2, label=f"YES: {word}")
        if close_dt:
            ax_price.axvline(close_dt, color="green", alpha=0.25, linewidth=0.8, linestyle="--")

    for m, df in sorted(no_markets, key=lambda x: x[0].get("volume", 0), reverse=True):
        word = m.get("yes_sub_title", m["ticker"])
        ax_price.plot(df["ts"], df["mid"], color="red", alpha=0.35, linewidth=0.9)

    ax_price.set_ylabel("Yes Price (cents = % probability)", fontsize=10)
    ax_price.set_ylim(-2, 102)
    ax_price.axhline(50, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))
    ax_price.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 10)))
    plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax_price.set_title("Market prices over speech (green=YES outcome, red=NO outcome; dashed lines=word said)",
                        fontsize=9)

    # Annotate YES resolutions
    for m, df in yes_markets:
        word = m.get("yes_sub_title", "")
        close_dt = iso_to_dt(m.get("close_time"))
        if close_dt:
            ax_price.text(close_dt, 102, word[:6], fontsize=5, color="darkgreen",
                          ha="center", va="bottom", rotation=90)

    # ── Panel 2: Volume per minute ─────────────────────────────────────────
    all_dfs = [df.set_index("ts")[["volume"]] for _, df in market_data.values()]
    vol_combined = pd.concat(all_dfs).resample("1min").sum()
    ax_vol.bar(vol_combined.index, vol_combined["volume"], width=pd.Timedelta("55s"),
               color="steelblue", alpha=0.7)
    ax_vol.set_ylabel("Total contracts/min", fontsize=9)
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))
    ax_vol.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 10)))
    plt.setp(ax_vol.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax_vol.set_title("Total volume across all markets (1 contract = 1 cent)", fontsize=9)

    # ── Panel 3: Reaction time ─────────────────────────────────────────────
    # For each YES market: how many minutes before close_time did price cross 80%?
    reaction_times = []
    for m, df in yes_markets:
        close_dt = iso_to_dt(m.get("close_time"))
        if close_dt is None or len(df) < 2:
            continue
        close_dt_aware = pd.Timestamp(close_dt)
        before = df[df["ts"] < close_dt_aware].copy()
        if before.empty:
            continue
        crossed = before[before["mid"] >= 80]
        if crossed.empty:
            # never hit 80% before resolution
            first_cross_min = None
        else:
            first_cross_ts = crossed["ts"].iloc[0]
            mins_before = (close_dt_aware - first_cross_ts).total_seconds() / 60
            reaction_times.append(mins_before)

    if reaction_times:
        ax_react.hist(reaction_times, bins=15, color="mediumseagreen", edgecolor="white")
        ax_react.axvline(np.median(reaction_times), color="darkgreen", linestyle="--",
                         label=f"Median {np.median(reaction_times):.0f} min")
        ax_react.set_xlabel("Minutes before word said (when price first hit 80%)", fontsize=8)
        ax_react.set_ylabel("# markets", fontsize=8)
        ax_react.set_title("Reaction time: how early did price go above 80%?", fontsize=9)
        ax_react.legend(fontsize=8)
    else:
        ax_react.text(0.5, 0.5, "Not enough YES data", ha="center", va="center",
                      transform=ax_react.transAxes)

    # ── Panel 4: Bid-ask spread distribution ──────────────────────────────
    all_spreads = pd.concat([df["spread"].dropna() for _, df in market_data.values()])
    ax_spread.hist(all_spreads[all_spreads > 0], bins=20, color="coral", edgecolor="white")
    ax_spread.set_xlabel("Bid-ask spread (cents)", fontsize=8)
    ax_spread.set_ylabel("# candles", fontsize=8)
    ax_spread.set_title("Bid-ask spread distribution (your round-trip cost per contract)", fontsize=9)
    median_spread = all_spreads[all_spreads > 0].median()
    ax_spread.axvline(median_spread, color="red", linestyle="--",
                      label=f"Median {median_spread:.0f}¢")
    ax_spread.legend(fontsize=8)

    # ── Transcript overlay ────────────────────────────────────────────────
    if transcript_path and os.path.exists(transcript_path):
        with open(transcript_path, encoding="utf-8") as f:
            text = f.read()
        words = text.split()
        total_words = len(words)
        speech_duration = (speech_end - speech_start).total_seconds()

        # Estimate: Trump speaks ~130 words/minute
        wpm = 130
        est_duration_s = (total_words / wpm) * 60

        # Find YES words in transcript and mark estimated timing
        for m, df in yes_markets:
            word = m.get("yes_sub_title", "").lower()
            close_dt = iso_to_dt(m.get("close_time"))
            if not word or not close_dt:
                continue
            # Find rough word position in transcript
            word_lower = text.lower()
            pos = word_lower.find(word.split("/")[0].strip())
            if pos > 0:
                frac = pos / len(text)
                est_ts = speech_start + pd.Timedelta(seconds=frac * est_duration_s)
                ax_price.axvline(est_ts, color="orange", alpha=0.3, linewidth=1, linestyle=":")

        print(f"Transcript loaded: {total_words} words (~{total_words//wpm} min estimated)")
        print(f"Orange dotted lines = estimated position in speech based on transcript word order\n")

    # ── Print analysis summary ────────────────────────────────────────────
    print("--- LIQUIDITY ANALYSIS -------------------------------------------")
    total_vol = sum(df["volume"].sum() for _, df in market_data.values())
    total_vol_dollars = total_vol / 100  # cents to dollars
    print(f"Total volume (all markets):  {total_vol:,.0f} contracts  ≈ ${total_vol_dollars:,.0f}")
    avg_vol_per_market = total_vol / len(market_data)
    print(f"Avg volume per market:       {avg_vol_per_market:,.0f} contracts  ≈ ${avg_vol_per_market/100:,.0f}")
    print(f"Median bid-ask spread:       {median_spread:.0f} cents  (round-trip cost per contract)")
    print(f"  → If you trade $100 at market price, you pay ~${median_spread:.0f} in spread")

    print("\n--- PRICE PREDICTIVENESS ------------------------------------------")
    # For YES markets: price at various points before resolution
    checkpoints = [60, 30, 15, 5, 2, 0]
    rows = []
    for m, df in yes_markets:
        close_dt = pd.Timestamp(iso_to_dt(m.get("close_time")))
        word = m.get("yes_sub_title","")
        row = {"word": word}
        for mins in checkpoints:
            cutoff = close_dt - pd.Timedelta(minutes=mins)
            before = df[df["ts"] <= cutoff]
            if before.empty:
                row[f"T-{mins}m"] = None
            else:
                row[f"T-{mins}m"] = before.iloc[-1]["mid"]
        rows.append(row)

    if rows:
        df_summary = pd.DataFrame(rows)
        cols = ["word"] + [f"T-{m}m" for m in checkpoints]
        print(df_summary[cols].to_string(index=False))
        print()
        print("Columns = price (cents) at X minutes before the word was said.")
        print("If T-30m ≈ T-0m, the market already priced it in early (hard to exploit).")
        print("If T-5m < T-0m by a lot, there's a fast move you might catch.\n")

    print("\n--- VIABILITY SUMMARY ---------------------------------------------")
    if reaction_times:
        pct_early = sum(1 for r in reaction_times if r > 5) / len(reaction_times) * 100
        print(f"  YES markets where price hit 80% >5 min before resolution: {pct_early:.0f}%")
        print(f"  Median time above 80% before word said: {np.median(reaction_times):.0f} min")
    print(f"  Median spread: {median_spread:.0f}¢  → need >spread¢ of edge to profit")
    print(f"  Total $ volume available: ~${total_vol_dollars:,.0f} across {len(market_data)} markets")
    print(f"  Max realistic bet size per market: ~${avg_vol_per_market/100/10:,.0f}-${avg_vol_per_market/100/5:,.0f}")
    print()
    print("  KEY QUESTIONS TO ANSWER BEFORE TRADING:")
    print("  1. Does your model see the word coming 5+ min early?")
    print("     If prices only jump when the word is SAID, the market is efficient and")
    print("     you can't beat it without faster-than-human reaction time.")
    print("  2. Can you get filled? At $10-50 per bet, spread costs ~1-2%, feasible.")
    print("  3. Are you faster than existing algos? Check if price moves happen")
    print("     in bursts (algos reacting) or gradually (sentiment building).")
    print("  4. Live trading means latency — plan for 1-2 second delay from speech")
    print("     audio to your bet being placed.")
    print()

    plt.savefig(f"{event_ticker}_analysis.png", dpi=130, bbox_inches="tight")
    print(f"  Chart saved to {event_ticker}_analysis.png")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("event_ticker", nargs="?", help="e.g. KXTRUMPMENTION-26FEB28")
    parser.add_argument("--transcript", help="Path to a .txt transcript file")
    args = parser.parse_args()

    if not args.event_ticker:
        list_events()
    else:
        analyse(args.event_ticker, transcript_path=args.transcript)
