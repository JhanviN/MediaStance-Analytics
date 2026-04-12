#!/usr/bin/env python3
"""
STEP 2 (prep) — Build data/labeled_dataset.csv from a raw corpus CSV.

Adds an empty `label` column. Fill each row with exactly one of:
  cooperative | neutral | adversarial
(Wording only; unclear → neutral.)

Examples:
  python scripts/init_labeled_template.py --latest
  python scripts/init_labeled_template.py -i data/raw_headlines_20260412_040641.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

RAW_GLOB = "raw_headlines*.csv"
OUTPUT_DEFAULT = DATA / "labeled_dataset.csv"
LABEL_COL = "label"


def _find_latest_raw() -> Path:
    paths = sorted(DATA.glob(RAW_GLOB), key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths:
        raise FileNotFoundError(f"No {RAW_GLOB} under {DATA}")
    return paths[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Create labeled_dataset.csv template from raw CSV")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--latest", action="store_true", help=f"Use newest {RAW_GLOB} in data/")
    g.add_argument("-i", "--input", type=Path, help="Path to raw_headlines*.csv")
    ap.add_argument("-o", "--output", type=Path, default=OUTPUT_DEFAULT, help="Output path")
    ap.add_argument("--force", action="store_true", help="Overwrite output if it exists")
    args = ap.parse_args()

    src = _find_latest_raw() if args.latest else Path(args.input)
    if not src.exists():
        sys.exit(f"Input not found: {src}")

    out = Path(args.output)
    if out.exists() and not args.force:
        sys.exit(f"Refusing to overwrite {out} (use --force)")

    with open(src, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"No rows in {src}")

    fieldnames = list(rows[0].keys())
    if LABEL_COL not in fieldnames:
        fieldnames.append(LABEL_COL)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in fieldnames if k != LABEL_COL}
            row[LABEL_COL] = ""
            w.writerow(row)

    print(f"Source: {src} ({len(rows)} rows)")
    print(f"Wrote:  {out}")
    print(f"Next:   fill column '{LABEL_COL}' with: cooperative | neutral | adversarial")
    print("Then:    python scripts/split_data.py")


if __name__ == "__main__":
    main()
