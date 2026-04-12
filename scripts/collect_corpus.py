#!/usr/bin/env python3
"""
STEP 1 — Collect bilateral headlines from live RSS (no dummy text).

For each article, if BOTH countries of a target pair appear in the headline+summary,
emit one row per matching pair. Output: data/raw_headlines.csv

Columns: id, headline, country_1, country_2, source, url, published_at, text

Run from repository root:
  python scripts/collect_corpus.py
  python scripts/collect_corpus.py --feeds 50 --merge
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import feedparser

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.news_fetcher import (  # noqa: E402
    _parse_published,
    _strip_html,
    detect_countries,
    fetch_articles,
)
from nlp.corpus_pairs import CORPUS_TARGET_PAIRS  # noqa: E402
from nlp.supplemental_feeds import SUPPLEMENTAL_PAIR_FEEDS  # noqa: E402

DEFAULT_OUTPUT = ROOT / "data" / "raw_headlines.csv"
FIELDNAMES = [
    "id",
    "headline",
    "country_1",
    "country_2",
    "source",
    "url",
    "published_at",
    "text",
]


def _row_id(url: str, c1: str, c2: str) -> str:
    h = hashlib.sha256(f"{url}|{c1}|{c2}".encode("utf-8")).hexdigest()
    return h[:20]


def _append_row(
    rows: List[Dict[str, str]],
    seen_keys: Set[Tuple[str, str, str]],
    *,
    url: str,
    c1: str,
    c2: str,
    headline: str,
    text: str,
    source: str,
    published_at: str,
) -> None:
    key = (url, c1, c2)
    if key in seen_keys or not headline:
        return
    seen_keys.add(key)
    rows.append(
        {
            "id": _row_id(url, c1, c2),
            "headline": headline[:500],
            "country_1": c1,
            "country_2": c2,
            "source": source,
            "url": url,
            "published_at": published_at,
            "text": text[:4000].replace("\n", " "),
        }
    )


def _collect_supplemental_rows(max_per_feed: int) -> List[Dict[str, str]]:
    """Google News search RSS: each feed is already scoped to one bilateral pair."""
    rows: List[Dict[str, str]] = []
    seen_keys: Set[Tuple[str, str, str]] = set()
    for spec in SUPPLEMENTAL_PAIR_FEEDS:
        ca, cb = spec["pair"]
        c1, c2 = sorted([ca.upper(), cb.upper()])
        try:
            feed = feedparser.parse(spec["url"])
        except Exception as e:
            print(f"  [Warning] supplemental {spec['label']}: {e}")
            continue
        for entry in feed.entries[:max_per_feed]:
            title = (entry.get("title") or "").strip().replace("\n", " ")
            summary = _strip_html(str(entry.get("summary", "") or ""))
            text = f"{title}. {summary}"
            url = (entry.get("link") or "").strip() or f"urn:gn:{hash(text)}"
            pub = _parse_published(entry).strftime("%Y-%m-%dT%H:%M:%SZ")
            _append_row(
                rows,
                seen_keys,
                url=url,
                c1=c1,
                c2=c2,
                headline=title,
                text=text,
                source=spec["label"],
                published_at=pub,
            )
    return rows


def _collect_general_rows(max_per_feed: int, max_age_days: int | None) -> List[Dict[str, str]]:
    """General RSS: keep rows only when both countries of a target pair are detected."""
    articles = fetch_articles(max_per_feed=max_per_feed, max_age_days=max_age_days)
    seen_keys: Set[Tuple[str, str, str]] = set()
    rows: List[Dict[str, str]] = []

    for art in articles:
        text = art.get("text") or ""
        found = detect_countries(text)
        if len(found) < 2:
            continue
        found_set = set(found)
        url = art.get("url") or ""

        for ca, cb in CORPUS_TARGET_PAIRS:
            if ca not in found_set or cb not in found_set:
                continue
            c1, c2 = sorted([ca.upper(), cb.upper()])
            title = (art.get("title") or "").strip().replace("\n", " ")
            _append_row(
                rows,
                seen_keys,
                url=url,
                c1=c1,
                c2=c2,
                headline=title,
                text=text,
                source=art.get("source", ""),
                published_at=art.get("published", ""),
            )
    return rows


def _collect_rows(max_per_feed: int, max_age_days: int | None, skip_supplemental: bool) -> List[Dict[str, str]]:
    general = _collect_general_rows(max_per_feed, max_age_days)
    if skip_supplemental:
        return general
    sup = _collect_supplemental_rows(max_per_feed)
    seen = {(r["url"], r["country_1"], r["country_2"]) for r in general}
    for r in sup:
        k = (r["url"], r["country_1"], r["country_2"])
        if k not in seen:
            general.append(r)
            seen.add(k)
    return general


def _load_existing_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect bilateral RSS corpus (4 target pairs)")
    ap.add_argument("--feeds", type=int, default=35, help="Max entries per RSS feed")
    ap.add_argument(
        "--days",
        type=int,
        default=None,
        metavar="N",
        help="Include headlines up to N days old (default: core.config.MAX_ARTICLE_AGE_DAYS). "
        "Use 90–120 when a short window yields too few bilateral matches.",
    )
    ap.add_argument(
        "--no-gnews",
        action="store_true",
        help="Skip Google News pair RSS (only general wire RSS + keyword match)",
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        help="Keep existing rows; append only new ids from this fetch",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path",
    )
    args = ap.parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Target pairs ({len(CORPUS_TARGET_PAIRS)}): {CORPUS_TARGET_PAIRS}")
    print("Fetching live RSS…")
    new_rows = _collect_rows(
        max_per_feed=args.feeds,
        max_age_days=args.days,
        skip_supplemental=args.no_gnews,
    )

    if args.merge and out_path.exists():
        existing = _load_existing_rows(out_path)
        old_ids = {r["id"] for r in existing if r.get("id")}
        merged = list(existing)
        added = 0
        for r in new_rows:
            if r["id"] not in old_ids:
                merged.append(r)
                old_ids.add(r["id"])
                added += 1
        out_rows = merged
        print(f"Merge: {len(existing)} existing + {added} new → {len(out_rows)} total")
    else:
        out_rows = new_rows
        print(f"Collected {len(out_rows)} bilateral rows (this run only)")

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for row in out_rows:
            w.writerow({k: row.get(k, "") for k in FIELDNAMES})

    print(f"Wrote {out_path} ({len(out_rows)} rows)")
    by_pair: Dict[str, int] = {}
    for r in out_rows:
        k = f"{r['country_1']}-{r['country_2']}"
        by_pair[k] = by_pair.get(k, 0) + 1
    print("Counts by pair:", dict(sorted(by_pair.items())))


if __name__ == "__main__":
    main()
