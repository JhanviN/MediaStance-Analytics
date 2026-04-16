"""
Merge synthetic + GDELT data into labeled_dataset.csv.

Handles:
- Deduplication by id
- Class balancing (optional cap per class to avoid synthetic domination)
- Preserves all existing human labels
- Outputs a merged labeled_dataset.csv ready for split_data.py

Usage:
    # After running both fetch_gdelt.py and generate_synthetic.py:
    python scripts/merge_augmented_data.py

    # With explicit paths:
    python scripts/merge_augmented_data.py \\
        --synthetic data/synthetic_raw.csv \\
        --gdelt data/gdelt_raw.csv \\
        --human data/labeled_dataset.csv \\
        --out data/labeled_dataset_augmented.csv \\
        --cap-per-class 600
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"

FIELDNAMES = [
    "id", "headline", "country_1", "country_2",
    "source", "url", "published_at", "text", "label",
]

VALID_LABELS = frozenset({"cooperative", "neutral", "adversarial"})


def _load(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if (r.get("label") or "").strip().lower() in VALID_LABELS]


def _is_placeholder(row: Dict) -> bool:
    """Detect unenriched GDELT rows that still have domain-only headlines."""
    h = (row.get("headline") or "").strip()
    return h.startswith("[GDELT") or len(h) < 15


def _load_gdelt(path: Path) -> List[Dict]:
    """Load GDELT rows, dropping unenriched placeholders."""
    rows = _load(path)
    before = len(rows)
    rows = [r for r in rows if not _is_placeholder(r)]
    dropped = before - len(rows)
    if dropped:
        print(f"  Dropped {dropped} unenriched GDELT placeholder rows (run --enrich to fix)")
    return rows


def _write(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in FIELDNAMES})


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge human + synthetic + GDELT data")
    ap.add_argument("--human", type=Path, default=DATA / "labeled_dataset.csv")
    ap.add_argument("--synthetic", type=Path, default=DATA / "synthetic_raw.csv")
    ap.add_argument("--gdelt", type=Path, default=DATA / "gdelt_raw.csv")
    ap.add_argument("--out", type=Path, default=DATA / "labeled_dataset_augmented.csv")
    ap.add_argument(
        "--cap-per-class", type=int, default=600,
        help="Max rows per label class (prevents synthetic data from dominating). "
             "Human labels are always kept. Default: 600"
    )
    ap.add_argument(
        "--gdelt-cap", type=int, default=300,
        help="Max GDELT rows per (pair, label) combo — GDELT labels are noisy. Default: 300"
    )
    ap.add_argument(
        "--no-gdelt", action="store_true",
        help="Skip GDELT data (use only human + synthetic)"
    )
    ap.add_argument(
        "--no-synthetic", action="store_true",
        help="Skip synthetic data (use only human + GDELT)"
    )
    args = ap.parse_args()

    # 1. Load all sources
    human_rows = _load(args.human)
    synth_rows = [] if args.no_synthetic else _load(args.synthetic)
    gdelt_rows = [] if args.no_gdelt else _load_gdelt(args.gdelt)

    print(f"Human labeled rows:    {len(human_rows)}")
    print(f"Synthetic rows:        {len(synth_rows)}")
    print(f"GDELT rows:            {len(gdelt_rows)}")

    # 2. Deduplicate — human rows take priority
    seen_ids: set = set()
    merged: List[Dict] = []

    # Always keep all human rows
    for r in human_rows:
        rid = (r.get("id") or "").strip()
        if rid and rid not in seen_ids:
            seen_ids.add(rid)
            merged.append(r)

    # 3. Cap GDELT per (pair, label) — it's noisy
    gdelt_by_pair_label: Dict[str, List[Dict]] = defaultdict(list)
    for r in gdelt_rows:
        key = f"{r.get('country_1','')}-{r.get('country_2','')}-{r.get('label','')}"
        gdelt_by_pair_label[key].append(r)

    gdelt_filtered: List[Dict] = []
    for key, rows in gdelt_by_pair_label.items():
        gdelt_filtered.extend(rows[:args.gdelt_cap])

    print(f"GDELT after per-(pair,label) cap of {args.gdelt_cap}: {len(gdelt_filtered)}")

    # 4. Add GDELT (deduplicated)
    for r in gdelt_filtered:
        rid = (r.get("id") or "").strip()
        if rid and rid not in seen_ids:
            seen_ids.add(rid)
            merged.append(r)

    # 5. Add synthetic (deduplicated)
    for r in synth_rows:
        rid = (r.get("id") or "").strip()
        if rid and rid not in seen_ids:
            seen_ids.add(rid)
            merged.append(r)

    print(f"\nBefore class cap: {len(merged)} total rows")
    print(f"Label dist: {dict(Counter(r['label'] for r in merged))}")

    # 6. Apply per-class cap (human rows are exempt)
    human_ids = {(r.get("id") or "").strip() for r in human_rows}
    label_counts: Counter = Counter()

    # Count human rows first (they're exempt from cap)
    for r in merged:
        if (r.get("id") or "").strip() in human_ids:
            label_counts[r["label"]] += 1

    final: List[Dict] = []
    for r in merged:
        rid = (r.get("id") or "").strip()
        lab = r.get("label", "")
        if rid in human_ids:
            # Always keep human rows
            final.append(r)
        elif label_counts[lab] < args.cap_per_class:
            label_counts[lab] += 1
            final.append(r)

    print(f"\nAfter class cap of {args.cap_per_class}: {len(final)} total rows")
    label_final = Counter(r["label"] for r in final)
    pair_final = Counter(f"{r.get('country_1','')}-{r.get('country_2','')}" for r in final)
    print(f"Label dist: {dict(label_final)}")
    print(f"Pair dist:  {dict(pair_final)}")

    # 7. Write output
    _write(args.out, final)
    print(f"\nWrote {len(final)} rows → {args.out}")
    print(f"\nNext steps:")
    print(f"  python scripts/split_data.py --input {args.out}")
    print(f"  python scripts/train_baseline.py")
    print(f"  python scripts/train_transformer.py")


if __name__ == "__main__":
    main()
