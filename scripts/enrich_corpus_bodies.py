

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import trafilatura
except ImportError:
    print("Install: pip install trafilatura", file=sys.stderr)
    raise SystemExit(1)

from core.news_fetcher import build_article_text, clean_rss_text  # noqa: E402


def _fetch_body(url: str, timeout: int) -> str:
    if not url or url.startswith("urn:"):
        return ""
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=False)
        if not downloaded:
            return ""
        return (trafilatura.extract(downloaded, include_comments=False, include_tables=False) or "").strip()
    except Exception:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich CSV text column from article URLs")
    ap.add_argument("-i", "--input", type=Path, required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="Max rows to fetch (0 = all)")
    ap.add_argument("--sleep", type=float, default=0.8, help="Seconds between HTTP requests")
    ap.add_argument("--timeout", type=int, default=25)
    args = ap.parse_args()

    with open(args.input, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys()) if rows else []

    fetched = 0
    for idx, row in enumerate(rows):
        if args.limit and idx >= args.limit:
            break
        url = (row.get("url") or "").strip()
        headline = clean_rss_text(row.get("headline", ""))
        body = _fetch_body(url, args.timeout)
        fetched += 1
        if body:
            snippet = body[:8000].replace("\n", " ")
            row["text"] = (
                build_article_text(headline, snippet) if headline else snippet[:4000]
            )
        time.sleep(args.sleep)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"Rows processed (HTTP): {fetched} | all rows written: {len(rows)} → {args.output}")


if __name__ == "__main__":
    main()
