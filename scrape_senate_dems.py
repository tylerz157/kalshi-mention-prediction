"""
Scraper for Senate Democrats Trump Transcripts
Source: https://www.democrats.senate.gov/newsroom/trump-transcripts
16 pages, ~300 transcripts

Structure (confirmed from HTML inspection):
- Index page: transcript links use class="ArticleTitle"
- Transcript page: content in div.js-press-release
  - Starts after <p>[Video]</p>
  - Ends before <p>Transcript courtesy of CQ Factbase.</p>
"""

import argparse
import requests
from bs4 import BeautifulSoup
import time
import os
import json
from urllib.parse import urljoin

BASE_URL = "https://www.democrats.senate.gov"
INDEX_URL = "https://www.democrats.senate.gov/newsroom/trump-transcripts"
OUTPUT_DIR = "transcripts/senate_dems"
FAILED_LOG = "transcripts/senate_dems_failed.json"
TOTAL_PAGES = 16
DELAY = 1.5  # seconds between requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def get_page(url, session, retries=3):
    for attempt in range(retries):
        try:
            resp = session.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            print(f"  Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(3)
    return None


def get_transcript_links(page_num, session):
    """Extract all transcript links from a single index page."""
    url = f"{INDEX_URL}?pagenum_rs={page_num}"
    print(f"Fetching index page {page_num}: {url}")
    resp = get_page(url, session)
    if not resp:
        print(f"  Failed to fetch index page {page_num}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # Links are <a class="ArticleTitle" href="...">
    links = []
    for a in soup.find_all("a", class_="ArticleTitle"):
        href = a.get("href", "")
        if href:
            full_url = urljoin(BASE_URL, href)
            if full_url not in links:
                links.append(full_url)

    print(f"  Found {len(links)} transcript links")
    return links


def extract_transcript(soup):
    """
    Extract transcript text from a transcript page.

    Content is in div.js-press-release. We collect all <p> tags,
    skip the opening [Video] paragraph, and stop before the
    'Transcript courtesy of CQ Factbase' paragraph.
    """
    content_div = soup.find("div", class_="js-press-release")
    if not content_div:
        return ""

    paragraphs = content_div.find_all("p")

    lines = []
    recording = False

    for p in paragraphs:
        text = p.get_text(separator=" ", strip=True)

        # Start recording after the [Video] paragraph
        if not recording:
            if "[Video]" in text or p.find("a", href=lambda h: h and "vimeo" in h):
                recording = True
            continue

        # Stop at the Transcript courtesy footer
        if "Transcript courtesy of CQ Factbase" in text:
            break

        if text:
            lines.append(text)

    return "\n\n".join(lines)


def get_title(soup):
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else "Unknown Title"


def slug_from_url(url):
    return url.rstrip("/").split("/")[-1]


def save_transcript(slug, title, url, transcript_text):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, f"{slug}.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"TITLE: {title}\n")
        f.write(f"URL: {url}\n")
        f.write(f"SOURCE: Senate Democrats / CQ Factbase\n")
        f.write("=" * 80 + "\n\n")
        f.write(transcript_text)

    return filepath


def fetch_and_save(urls, session, existing, delay):
    """Fetch a list of transcript URLs and save them. Returns list of failed URLs."""
    failed = []
    for i, url in enumerate(urls, 1):
        slug = slug_from_url(url)

        if slug in existing:
            print(f"[{i}/{len(urls)}] Skipping (already saved): {slug}")
            continue

        print(f"[{i}/{len(urls)}] Fetching: {slug}")
        resp = get_page(url, session)

        if not resp:
            print(f"  FAILED: {url}")
            failed.append(url)
            time.sleep(delay)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        title = get_title(soup)
        transcript_text = extract_transcript(soup)

        if len(transcript_text) < 100:
            print(f"  WARNING: Very short transcript ({len(transcript_text)} chars)")

        filepath = save_transcript(slug, title, url, transcript_text)
        print(f"  Saved: {filepath} ({len(transcript_text)} chars)")

        existing.add(slug)
        time.sleep(delay)

    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry only the URLs in the failed log instead of a full scrape",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    existing = set(
        f.replace(".txt", "") for f in os.listdir(OUTPUT_DIR) if f.endswith(".txt")
    )
    if existing:
        print(f"Found {len(existing)} transcripts already downloaded.\n")

    with requests.Session() as session:
        if args.retry_failed:
            if not os.path.exists(FAILED_LOG):
                print("No failed log found. Nothing to retry.")
                return

            with open(FAILED_LOG) as f:
                urls_to_retry = json.load(f)

            print(f"=== Retrying {len(urls_to_retry)} failed URLs (longer delays) ===\n")
            # Use longer delay and more retries for bot-detected requests
            failed = fetch_and_save(urls_to_retry, session, existing, delay=5)

        else:
            # Phase 1: collect all transcript URLs
            print("=== Phase 1: Collecting transcript URLs ===")
            all_links = []
            for page_num in range(1, TOTAL_PAGES + 1):
                links = get_transcript_links(page_num, session)
                all_links.extend(links)
                time.sleep(DELAY)

            all_links = list(dict.fromkeys(all_links))
            print(f"\nTotal unique transcript URLs found: {len(all_links)}\n")

            # Phase 2: fetch and save
            print("=== Phase 2: Fetching transcripts ===")
            failed = fetch_and_save(all_links, session, existing, delay=DELAY)

    if failed:
        with open(FAILED_LOG, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"\nFailed URLs ({len(failed)}) saved to {FAILED_LOG}")
    elif args.retry_failed and os.path.exists(FAILED_LOG):
        os.remove(FAILED_LOG)
        print("\nAll retries succeeded. Removed failed log.")

    print(f"\nDone. {len(existing)} transcripts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()