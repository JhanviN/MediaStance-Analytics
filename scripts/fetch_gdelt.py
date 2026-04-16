"""
GDELT 2.0 real-data fetcher for bilateral pair labeling.

Pulls from GDELT GKG/Events CSV (no API key needed — public BigQuery-style CSVs).
Maps CAMEO event codes → adversarial / cooperative / neutral.

Usage:
    python scripts/fetch_gdelt.py --days 30 --out data/gdelt_raw.csv
    python scripts/fetch_gdelt.py --days 90 --merge --out data/gdelt_raw.csv

Output schema matches raw_headlines.csv so sync_labeled_dataset.py works directly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.corpus_pairs import CORPUS_TARGET_PAIRS  # noqa: E402

# ── CAMEO code → label mapping ────────────────────────────────────────────────
# CAMEO root codes: https://parusanalytics.com/eventdata/cameo.dir/CAMEO.Manual.1.1b3.pdf
# Cooperative: 01-08 (verbal/material cooperation)
# Neutral:     09-11 (consult, demand, disapprove — ambiguous)
# Adversarial: 12-20 (reject, threaten, protest, coerce, assault, fight, mass violence)

def cameo_to_label(code: str) -> Optional[str]:
    """Map CAMEO EventRootCode to our 3-class label. Returns None to skip."""
    if not code:
        return None
    try:
        root = int(str(code).strip()[:2])
    except ValueError:
        return None
    if root in (1, 2, 3, 4, 5, 6, 7, 8):
        return "cooperative"
    if root in (9, 10, 11):
        return "neutral"
    if root in (12, 13, 14, 15, 16, 17, 18, 19, 20):
        return "adversarial"
    return None


# ── Country code mapping (GDELT Geo fields use ISO 3166-1 alpha-3) ───────────
GDELT_TO_ISO2: Dict[str, str] = {
    "IND": "IN",   # India
    "CHN": "CN",   # China
    "USA": "US",   # United States
    "RUS": "RU",   # Russia
    "PAK": "PK",   # Pakistan
    "IRN": "IR",   # Iran
    "ISR": "IL",   # Israel
}

TARGET_PAIRS: List[Tuple[str, str]] = [
    tuple(sorted([a, b])) for a, b in CORPUS_TARGET_PAIRS  # type: ignore[misc]
]

FIELDNAMES = [
    "id", "headline", "country_1", "country_2",
    "source", "url", "published_at", "text", "label",
]


def _row_id(url: str, c1: str, c2: str) -> str:
    h = hashlib.sha256(f"gdelt|{url}|{c1}|{c2}".encode()).hexdigest()
    return h[:20]


def _gdelt_15min_urls(days: int) -> List[str]:
    """
    Build GDELT 2.0 export file URLs for the last `days` days.
    Instead of parsing the massive masterfilelist (millions of lines),
    we generate URLs directly from timestamps — GDELT publishes every 15 min.
    Format: http://data.gdeltproject.org/gdeltv2/YYYYMMDDHHMMSS.export.CSV.zip
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    urls = []
    # Walk backwards in 15-min steps
    t = now.replace(second=0, microsecond=0)
    # Round down to nearest 15-min boundary
    t = t.replace(minute=(t.minute // 15) * 15)

    while t >= cutoff:
        fname = t.strftime("%Y%m%d%H%M%S") + ".export.CSV.zip"
        urls.append(f"http://data.gdeltproject.org/gdeltv2/{fname}")
        t -= timedelta(minutes=15)

    urls.reverse()  # chronological order
    print(f"Generated {len(urls)} GDELT file URLs for last {days} days.")
    return urls


def _parse_gdelt_zip(url: str) -> List[Dict]:
    """Download one GDELT export zip and extract matching bilateral rows.

    GDELT 2.0 export columns (0-based, tab-separated, no header):
      0  = GlobalEventID
      1  = Day (YYYYMMDD)
      5  = Actor1Code (type code, e.g. 'EDU', 'GOV')
      7  = Actor1Geo_CountryCode  ← actual ISO2 country
      15 = Actor2Code
      17 = Actor2Geo_CountryCode  ← actual ISO2 country
      28 = EventRootCode (2-digit CAMEO root)
      60 = SOURCEURL
    """
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [skip] {url.split('/')[-1]}: {e}")
        return []

    rows = []
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".CSV")][0]
            with zf.open(csv_name) as f:
                reader = csv.reader(
                    io.TextIOWrapper(f, encoding="utf-8", errors="ignore"),
                    delimiter="\t",
                )
                # Collect all events first, then deduplicate by (url, pair)
                # keeping the most adversarial label (adversarial > neutral > cooperative)
                LABEL_PRIORITY = {"adversarial": 2, "neutral": 1, "cooperative": 0}
                url_pair_best: Dict[Tuple, Dict] = {}

                for row in reader:
                    if len(row) < 61:
                        continue

                    day_str = row[1].strip()
                    a1_raw = row[7].strip().upper()   # Actor1Geo_CountryCode (ISO3)
                    a2_raw = row[17].strip().upper()  # Actor2Geo_CountryCode (ISO3)
                    event_root = row[28].strip()      # EventRootCode (2-digit)
                    source_url = row[60].strip()

                    a1 = GDELT_TO_ISO2.get(a1_raw)
                    a2 = GDELT_TO_ISO2.get(a2_raw)
                    if not a1 or not a2 or a1 == a2:
                        continue

                    pair = tuple(sorted([a1, a2]))
                    if pair not in TARGET_PAIRS:
                        continue

                    label = cameo_to_label(event_root)
                    if label is None:
                        continue

                    if not source_url or not source_url.startswith("http"):
                        continue

                    try:
                        pub_dt = datetime.strptime(day_str, "%Y%m%d").replace(tzinfo=timezone.utc)
                        pub_str = pub_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                    except ValueError:
                        pub_str = ""

                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(source_url).netloc.replace("www.", "")
                    except Exception:
                        domain = ""

                    key = (source_url, pair)
                    existing = url_pair_best.get(key)
                    if existing is None or LABEL_PRIORITY[label] > LABEL_PRIORITY[existing["label"]]:
                        url_pair_best[key] = {
                            "id": _row_id(source_url, pair[0], pair[1]),
                            "headline": f"[GDELT {pair[0]}-{pair[1]} {label}] {domain}"[:500],
                            "country_1": pair[0],
                            "country_2": pair[1],
                            "source": f"GDELT-{event_root}",
                            "url": source_url,
                            "published_at": pub_str,
                            "text": f"[GDELT {pair[0]}-{pair[1]} {label}] {domain}"[:4000],
                            "label": label,
                        }

                rows = list(url_pair_best.values())
    except Exception as e:
        print(f"  [parse error] {url.split('/')[-1]}: {e}")

    return rows


def _enrich_headlines(rows: List[Dict], limit: int = 500) -> List[Dict]:
    """
    Fetch actual article titles from source URLs using requests + basic parsing.
    Only enriches rows where headline is a GDELT placeholder.
    Skips on any error (best-effort).
    """
    try:
        from bs4 import BeautifulSoup
        has_bs4 = True
    except ImportError:
        has_bs4 = False

    enriched = 0
    for row in rows:
        if enriched >= limit:
            break
        if not row["url"].startswith("http"):
            continue
        if not row["headline"].startswith("[GDELT"):
            continue
        try:
            r = requests.get(row["url"], timeout=5, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                continue
            if has_bs4:
                soup = BeautifulSoup(r.text, "html.parser")
                title = soup.find("title")
                if title and title.text.strip():
                    row["headline"] = title.text.strip()[:500]
                    row["text"] = title.text.strip()[:4000]
            else:
                # Fallback: crude regex
                import re
                m = re.search(r"<title[^>]*>([^<]+)</title>", r.text, re.IGNORECASE)
                if m:
                    row["headline"] = m.group(1).strip()[:500]
                    row["text"] = m.group(1).strip()[:4000]
            enriched += 1
            time.sleep(0.3)
        except Exception:
            continue

    print(f"Enriched {enriched} headlines from source URLs.")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch GDELT bilateral events → labeled CSV")
    ap.add_argument("--days", type=int, default=30, help="How many days back to fetch (default 30)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "gdelt_raw.csv")
    ap.add_argument("--merge", action="store_true", help="Merge with existing file (keep old rows)")
    ap.add_argument("--enrich", action="store_true", help="Fetch actual article titles from URLs (slow)")
    ap.add_argument("--enrich-limit", type=int, default=300, help="Max URLs to enrich (default 300)")
    ap.add_argument("--max-files", type=int, default=0, help="Limit number of GDELT files to process (0=all)")
    ap.add_argument("--probe", action="store_true", help="Test mode: process 3 files and print sample rows, don't write output")
    args = ap.parse_args()

    urls = _gdelt_15min_urls(args.days)
    if not urls:
        print("No GDELT files found. Check your internet connection.")
        sys.exit(1)

    if args.max_files > 0:
        urls = urls[:args.max_files]
        print(f"Processing first {len(urls)} files (--max-files limit).")

    # Probe mode: test 3 files and show sample
    if args.probe:
        print("\n[PROBE MODE] Testing 3 files...")
        probe_urls = urls[-3:]  # most recent 3
        for url in probe_urls:
            fname = url.split("/")[-1]
            print(f"\n  {fname}")
            batch = _parse_gdelt_zip(url)
            print(f"  → {len(batch)} matching rows")
            for r in batch[:3]:
                print(f"     {r['country_1']}-{r['country_2']} | {r['label']} | {r['url'][:60]}")
        return

    all_rows: List[Dict] = []
    seen_ids: set = set()

    # Load existing if merging
    if args.merge and args.out.exists():
        with open(args.out, newline="", encoding="utf-8-sig") as f:
            existing = list(csv.DictReader(f))
        for r in existing:
            if r.get("id"):
                seen_ids.add(r["id"])
        all_rows.extend(existing)
        print(f"Loaded {len(existing)} existing rows from {args.out.name}")

    new_count = 0
    for i, url in enumerate(urls):
        fname = url.split("/")[-1]
        print(f"[{i+1}/{len(urls)}] {fname}", end=" ... ", flush=True)
        batch = _parse_gdelt_zip(url)
        added = 0
        for row in batch:
            if row["id"] not in seen_ids:
                seen_ids.add(row["id"])
                all_rows.append(row)
                added += 1
        print(f"{added} new rows")
        new_count += added
        time.sleep(0.1)  # be polite to GDELT servers

    if args.enrich:
        unenriched = [r for r in all_rows if r.get("headline", "").startswith("[GDELT")]
        print(f"Enriching up to {args.enrich_limit} of {len(unenriched)} placeholder headlines...")
        all_rows = _enrich_headlines(all_rows, limit=args.enrich_limit)

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row.get(k, "") for k in FIELDNAMES})

    print(f"\nDone. {new_count} new rows added. Total: {len(all_rows)} rows → {args.out}")

    # Summary
    from collections import Counter
    labels = [r.get("label", "") for r in all_rows if r.get("label")]
    pairs = [f"{r.get('country_1','')}-{r.get('country_2','')}" for r in all_rows]
    print(f"Label dist:  {dict(Counter(labels))}")
    print(f"Pair dist:   {dict(Counter(pairs))}")


if __name__ == "__main__":
    main()
