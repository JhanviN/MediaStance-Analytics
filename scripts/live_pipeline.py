"""
MediaStance Analytics — Live Pipeline
Runs continuously: collect RSS → predict → save to DB → repeat every N minutes.

Usage:
    python scripts/live_pipeline.py                    # runs every 60 min
    python scripts/live_pipeline.py --interval 30      # every 30 min
    python scripts/live_pipeline.py --once             # single run, then exit
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.news_fetcher import fetch_articles, detect_countries, build_article_text
from nlp.corpus_pairs import CORPUS_TARGET_PAIRS
from nlp.supplemental_feeds import SUPPLEMENTAL_PAIR_FEEDS
from nlp.inference import predict_baseline, predict_transformer
from nlp.predictions_sqlite import connect, init_predictions_db, insert_prediction
from nlp.pair_utils import pair_key_from_codes
import feedparser
import hashlib


def _row_id(url: str, c1: str, c2: str) -> str:
    return hashlib.sha256(f"{url}|{c1}|{c2}".encode()).hexdigest()[:20]


def collect_live_headlines() -> list[dict]:
    """Fetch fresh headlines from RSS + Google News supplemental feeds."""
    from core import config
    from core.news_fetcher import clean_rss_text, _parse_published
    from datetime import timedelta

    headlines = []
    seen = set()
    target_pairs = set(tuple(sorted([a, b])) for a, b in CORPUS_TARGET_PAIRS)
    cutoff = datetime.now(timezone.utc) - timedelta(days=1)  # last 24h only

    # General RSS feeds
    articles = fetch_articles(max_per_feed=20, max_age_days=1)
    for art in articles:
        found = set(detect_countries(art.get("text", "")))
        for ca, cb in target_pairs:
            if ca in found and cb in found:
                c1, c2 = sorted([ca, cb])
                url = art.get("url", "")
                key = (url, c1, c2)
                if key not in seen:
                    seen.add(key)
                    headlines.append({
                        "headline": art.get("title", "")[:500],
                        "text": art.get("text", "")[:2000],
                        "country_1": c1,
                        "country_2": c2,
                        "url": url,
                        "published_at": art.get("published", ""),
                        "source": art.get("source", "RSS"),
                    })

    # Supplemental pair-specific feeds
    for spec in SUPPLEMENTAL_PAIR_FEEDS:
        ca, cb = spec["pair"]
        c1, c2 = sorted([ca.upper(), cb.upper()])
        try:
            feed = feedparser.parse(spec["url"])
            for entry in feed.entries[:15]:
                pub = _parse_published(entry)
                if pub < cutoff:
                    continue
                title = clean_rss_text(entry.get("title", ""))
                text = build_article_text(entry.get("title", ""), entry.get("summary", ""))
                url = (entry.get("link") or "").strip()
                key = (url, c1, c2)
                if key not in seen and title:
                    seen.add(key)
                    headlines.append({
                        "headline": title[:500],
                        "text": text[:2000],
                        "country_1": c1,
                        "country_2": c2,
                        "url": url,
                        "published_at": pub.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "source": spec["label"],
                    })
        except Exception as e:
            print(f"  [warn] {spec['label']}: {e}")

    return headlines


def run_once(conn, model: str = "baseline") -> int:
    """Collect, predict, save. Returns number of new predictions added."""
    print(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC] Collecting headlines...")
    headlines = collect_live_headlines()
    print(f"  Collected {len(headlines)} fresh headlines")

    if not headlines:
        print("  No new headlines found.")
        return 0

    added = 0
    for h in headlines:
        c1 = h["country_1"]
        c2 = h["country_2"]
        text = h["text"] or h["headline"]
        headline = h["headline"]
        input_text = f"{c1}-{c2}: {text}"

        try:
            if model in ("baseline", "both"):
                lab, conf, probs = predict_baseline(input_text)
                insert_prediction(
                    conn,
                    headline=headline,
                    text_used=input_text[:2000],
                    country_1=c1,
                    country_2=c2,
                    model="baseline",
                    label=lab,
                    confidence=conf,
                    probs=probs,
                )
                added += 1

            if model in ("transformer", "both"):
                lab, conf, probs = predict_transformer(input_text)
                insert_prediction(
                    conn,
                    headline=headline,
                    text_used=input_text[:2000],
                    country_1=c1,
                    country_2=c2,
                    model="transformer",
                    label=lab,
                    confidence=conf,
                    probs=probs,
                )
                added += 1

        except Exception as e:
            print(f"  [skip] {e}")
            continue

    print(f"  Saved {added} new predictions to DB")
    return added


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="MediaStance live pipeline")
    ap.add_argument("--interval", type=int, default=60, help="Minutes between runs (default 60)")
    ap.add_argument("--model", choices=["baseline", "transformer", "both"], default="baseline")
    ap.add_argument("--once", action="store_true", help="Run once and exit")
    ap.add_argument("--db", type=Path, default=ROOT / "data" / "predictions.db")
    args = ap.parse_args()

    conn = connect(args.db)
    init_predictions_db(conn)

    print(f"MediaStance Analytics — Live Pipeline")
    print(f"Model: {args.model} | Interval: {args.interval} min | DB: {args.db}")
    print(f"Pairs: {len(CORPUS_TARGET_PAIRS)} | Feeds: {len(SUPPLEMENTAL_PAIR_FEEDS)}")

    if args.once:
        run_once(conn, model=args.model)
        conn.close()
        return

    print(f"\nRunning continuously every {args.interval} minutes. Ctrl+C to stop.\n")
    while True:
        try:
            run_once(conn, model=args.model)
            print(f"  Next run in {args.interval} minutes...")
            time.sleep(args.interval * 60)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"  [error] {e} — retrying in 5 min")
            time.sleep(300)

    conn.close()


if __name__ == "__main__":
    main()
