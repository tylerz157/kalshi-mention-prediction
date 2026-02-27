"""
Fetch historical Kalshi market data for Trump mention markets.

Auth: RSA private key + key ID (from ./key file)
Output: data/kalshi/<ticker>_candlesticks.csv and <ticker>_trades.csv

Usage:
    python fetch_kalshi_markets.py
"""

import base64
import csv
import json
import os
import time

import requests
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# ── Config ──────────────────────────────────────────────────────────────────

KEY_FILE = "key"
BASE_URL = "https://api.elections.kalshi.com"
OUTPUT_DIR = "data/kalshi"

# Keywords used to identify mention-style markets in titles
MENTION_KEYWORDS = ["mention", "say ", "says", "word", "utter", "refer"]

# ── Auth ─────────────────────────────────────────────────────────────────────

def load_credentials(key_file):
    """Parse RSA private key and key ID from the key file."""
    with open(key_file) as f:
        contents = f.read()

    # Extract PEM block
    pem_start = contents.find("-----BEGIN RSA PRIVATE KEY-----")
    pem_end = contents.find("-----END RSA PRIVATE KEY-----") + len("-----END RSA PRIVATE KEY-----")
    pem = contents[pem_start:pem_end].strip()

    # Key ID is the UUID on its own line after the PEM block
    rest = contents[pem_end:].strip()
    key_id = rest.strip().splitlines()[-1].strip()

    private_key = serialization.load_pem_private_key(
        pem.encode(), password=None, backend=default_backend()
    )
    return key_id, private_key


def auth_headers(key_id, private_key, method, path):
    """Generate Kalshi RSA auth headers for a request."""
    timestamp_ms = str(int(time.time() * 1000))
    msg = timestamp_ms + method.upper() + path
    signature = private_key.sign(
        msg.encode("utf-8"),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        "Content-Type": "application/json",
    }


# ── API helpers ───────────────────────────────────────────────────────────────

def get(key_id, private_key, path, params=None):
    """Authenticated GET request. Returns parsed JSON or None on error."""
    url = BASE_URL + path
    headers = auth_headers(key_id, private_key, "GET", path)
    resp = requests.get(url, headers=headers, params=params, timeout=15)
    if resp.status_code != 200:
        print(f"  ERROR {resp.status_code} for {path}: {resp.text[:200]}")
        return None
    return resp.json()


# ── Market discovery ──────────────────────────────────────────────────────────

EVENTS_CHECKPOINT = "data/kalshi/trump_events_checkpoint.json"


def find_trump_events(key_id, private_key):
    """
    Page through settled events and return those whose ticker or title
    contains 'trump'. Saves a checkpoint after each page so the scan
    can resume if interrupted.
    """
    # Resume from checkpoint if it exists
    checkpoint = {"trump_events": [], "cursor": None, "page": 0, "done": False}
    if os.path.exists(EVENTS_CHECKPOINT):
        with open(EVENTS_CHECKPOINT) as f:
            checkpoint = json.load(f)
        if checkpoint.get("done"):
            print(f"Loaded {len(checkpoint['trump_events'])} Trump events from checkpoint (scan was complete).")
            return checkpoint["trump_events"]
        print(f"Resuming from page {checkpoint['page']} with {len(checkpoint['trump_events'])} events found so far...")

    path = "/trade-api/v2/events"
    trump_events = checkpoint["trump_events"]
    cursor = checkpoint["cursor"]
    page = checkpoint["page"]

    print("Searching settled events for Trump-related events...")
    while True:
        params = {"status": "settled", "limit": 200}
        if cursor:
            params["cursor"] = cursor

        data = get(key_id, private_key, path, params)
        if not data:
            break

        batch = data.get("events", [])
        page += 1

        for e in batch:
            ticker = (e.get("event_ticker") or "").lower()
            title = (e.get("title") or "").lower()
            combined = ticker + " " + title
            if "what will trump say during" in combined:
                trump_events.append(e)
                print(f"  Event: {e.get('event_ticker')} — {e.get('title')}")

        cursor = data.get("cursor")
        print(f"  Page {page}: scanned {len(batch)} events, {len(trump_events)} trump matches")

        # Save checkpoint after every page
        os.makedirs(os.path.dirname(EVENTS_CHECKPOINT), exist_ok=True)
        with open(EVENTS_CHECKPOINT, "w") as f:
            json.dump({"trump_events": trump_events, "cursor": cursor, "page": page, "done": not cursor}, f)

        if len(trump_events) >= 100:
            print(f"Reached 100 matches. Stopping early.")
            with open(EVENTS_CHECKPOINT, "w") as f:
                json.dump({"trump_events": trump_events, "cursor": cursor, "page": page, "done": True}, f)
            break

        if not cursor or not batch:
            break

    print(f"Scan complete. {len(trump_events)} events saved to {EVENTS_CHECKPOINT}")
    return trump_events


def find_trump_mention_markets(key_id, private_key):
    """
    Find Trump mention markets by first locating Trump events,
    then fetching their child markets and filtering for mention-style ones.
    """
    trump_events = find_trump_events(key_id, private_key)
    print(f"\nFound {len(trump_events)} Trump events. Fetching their markets...\n")

    mention_markets = []
    for event in trump_events:
        event_ticker = event.get("event_ticker")
        path = "/trade-api/v2/markets"
        data = get(key_id, private_key, path, {"event_ticker": event_ticker, "limit": 100})
        if not data:
            continue

        for m in data.get("markets", []):
            mention_markets.append(m)
            print(f"  Found: {m['ticker']} — {m['title']}")

    return mention_markets


# ── Data fetching ─────────────────────────────────────────────────────────────

def series_from_event(event_ticker):
    """Extract series ticker from event ticker, e.g. 'KXTRUMPMENTION-26FEB28' → 'KXTRUMPMENTION'."""
    return event_ticker.split("-")[0]


def fetch_candlesticks(key_id, private_key, ticker, event_ticker, open_time, close_time):
    """
    Fetch 1-minute candlesticks for a market.
    Correct path: /trade-api/v2/series/{series}/markets/{ticker}/candlesticks

    Uses a speech-window approach: start = max(open_time, close_time - 6h) to avoid
    hitting API range limits for markets that opened weeks before the speech.
    """
    series = series_from_event(event_ticker)
    path = f"/trade-api/v2/series/{series}/markets/{ticker}/candlesticks"

    close_ts = int(close_time.timestamp()) if hasattr(close_time, "timestamp") else close_time
    raw_open = int(open_time.timestamp()) if hasattr(open_time, "timestamp") else open_time
    # Cap start at 6 hours before close so we stay within the API's window limit
    start_ts = max(raw_open, close_ts - 6 * 3600)
    end_ts = close_ts + 600  # 10-min buffer after close

    params = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": 1,  # 1-minute candles
    }
    data = get(key_id, private_key, path, params)
    if not data:
        return []
    return data.get("candlesticks", [])


# ── Saving ────────────────────────────────────────────────────────────────────

def save_csv(rows, filepath, fieldnames):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows)} rows -> {filepath}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    key_id, private_key = load_credentials(KEY_FILE)
    print(f"Loaded credentials. Key ID: {key_id}\n")

    # Step 1: find Trump mention markets
    markets = find_trump_mention_markets(key_id, private_key)
    print(f"\nFound {len(markets)} Trump mention markets total.\n")

    if not markets:
        print("No markets found. Try broadening MENTION_KEYWORDS or check auth.")
        return

    # Save market metadata
    meta_path = os.path.join(OUTPUT_DIR, "trump_mention_markets.json")
    with open(meta_path, "w") as f:
        json.dump(markets, f, indent=2, default=str)
    print(f"Market metadata saved to {meta_path}\n")

    # Step 2: fetch candlesticks for each market (resumable — skips already saved)
    for i, m in enumerate(markets, 1):
        ticker = m["ticker"]
        out_path = os.path.join(OUTPUT_DIR, f"{ticker}_candlesticks.csv")
        if os.path.exists(out_path):
            print(f"[{i}/{len(markets)}] Skip (exists): {ticker}")
            continue
        print(f"[{i}/{len(markets)}] Fetching: {ticker} — {m.get('yes_sub_title', '')}")

        # Candlesticks
        open_ts = m.get("open_time", 0)
        close_ts = m.get("close_time", 0)

        # Convert ISO strings to unix timestamps if needed
        if isinstance(open_ts, str):
            from datetime import datetime, timezone
            open_ts = int(datetime.fromisoformat(open_ts.replace("Z", "+00:00")).timestamp())
            close_ts = int(datetime.fromisoformat(close_ts.replace("Z", "+00:00")).timestamp())

        event_ticker = m.get("event_ticker", "")
        candles = fetch_candlesticks(key_id, private_key, ticker, event_ticker, open_ts, close_ts)
        if candles:
            candle_fields = list(candles[0].keys())
            save_csv(candles, os.path.join(OUTPUT_DIR, f"{ticker}_candlesticks.csv"), candle_fields)
        else:
            print(f"  No candlestick data for {ticker}")

        print()
        time.sleep(0.5)

    print("Done.")


if __name__ == "__main__":
    main()
