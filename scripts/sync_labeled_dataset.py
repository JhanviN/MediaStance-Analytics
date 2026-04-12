#!/usr/bin/env python3
"""
Keep manual labels when new raw headlines are fetched.

- Reads latest (or chosen) raw_headlines*.csv
- If data/labeled_dataset.csv exists: for each raw row, keep existing label when id matches;
  new ids get an empty label; rows only in labeled (old id dropped from raw) stay at the end.
- If no labeled file yet: same as a fresh copy with empty labels.

Run after collect_corpus --merge so you never redo labeling for the same id.

Examples:
  python scripts/sync_labeled_dataset.py --latest
  python scripts/sync_labeled_dataset.py -i data/raw_headlines_20260412_040641.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_GLOB = "raw_headlines*.csv"
LABELED_DEFAULT = DATA / "labeled_dataset.csv"


def _find_latest_raw() -> Path:
    paths = sorted(DATA.glob(RAW_GLOB), key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths:
        raise FileNotFoundError(f"No {RAW_GLOB} under {DATA}")
    return paths[0]


def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        rows = list(r)
        fn = r.fieldnames or []
    return list(fn), rows


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> Path:
    try:
        _write_csv_impl(path, fieldnames, rows)
        return path
    except PermissionError:
        alt = DATA / f"labeled_dataset_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
        print(f"\n[Permission denied] {path}\n  Close the file in Excel/IDE, or using: {alt}\n", file=sys.stderr)
        _write_csv_impl(alt, fieldnames, rows)
        return alt


def _write_csv_impl(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def sync(raw_path: Path, labeled_path: Path) -> Tuple[int, int, int, Path]:
    """
    Returns: (n_raw, n_preserved_labels, n_orphans_kept, written_path)
    """
    raw_fields, raw_rows = _read_csv(raw_path)
    if "id" not in raw_fields:
        raise ValueError(f"Raw CSV missing id column: {raw_path}")

    label_key = "label"
    base_fields = [f for f in raw_fields if f != label_key]

    labeled_by_id: Dict[str, Dict[str, str]] = {}
    if labeled_path.exists():
        _lf, labeled_rows = _read_csv(labeled_path)
        for row in labeled_rows:
            rid = (row.get("id") or "").strip()
            if rid:
                labeled_by_id[rid] = dict(row)

    raw_by_id = {(r.get("id") or "").strip(): r for r in raw_rows if (r.get("id") or "").strip()}

    # Union fieldnames: raw columns + label + any extra keys from labeled rows
    fieldnames = list(dict.fromkeys(list(base_fields) + [label_key]))
    for row in labeled_by_id.values():
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)

    out: List[Dict[str, str]] = []
    preserved = 0

    # 1) One row per raw article, in raw file order — carry over label if known
    for r in raw_rows:
        rid = (r.get("id") or "").strip()
        merged = {k: (r.get(k) or "") for k in base_fields}
        merged["id"] = rid
        if rid in labeled_by_id:
            old = labeled_by_id[rid]
            lab = (old.get(label_key) or "").strip()
            merged[label_key] = lab
            if lab:
                preserved += 1
            for k, v in old.items():
                if k != label_key and k not in merged and v:
                    merged[k] = v
        else:
            merged[label_key] = ""
        for k in fieldnames:
            merged.setdefault(k, "")
        out.append(merged)

    # 2) Labeled rows whose id no longer appears in raw (keep so work is not lost)
    orphans = 0
    for rid, old in labeled_by_id.items():
        if rid not in raw_by_id:
            row = {k: "" for k in fieldnames}
            for k, v in old.items():
                if k in fieldnames:
                    row[k] = v
            out.append(row)
            orphans += 1

    written = _write_csv(labeled_path, fieldnames, out)
    return len(raw_rows), preserved, orphans, written


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge raw corpus into labeled_dataset.csv without losing labels")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--latest", action="store_true", help=f"Use newest {RAW_GLOB} in data/")
    g.add_argument("-i", "--input", type=Path, help="Path to raw_headlines*.csv")
    ap.add_argument("-o", "--output", type=Path, default=LABELED_DEFAULT, help="labeled_dataset.csv path")
    args = ap.parse_args()

    raw_path = _find_latest_raw() if args.latest else Path(args.input)
    if not raw_path.exists():
        sys.exit(f"Raw file not found: {raw_path}")

    n_raw, n_kept, n_orphan, written = sync(raw_path, args.output)
    print(f"Raw rows:           {n_raw}  ({raw_path.name})")
    print(f"Labels preserved:  {n_kept}  (same id had non-empty label)")
    print(f"Orphans kept:      {n_orphan}  (labeled id not in this raw pull — still in output)")
    print(f"Total output rows: {n_raw + n_orphan}")
    print(f"Wrote:             {written}")
    print("Fill empty `label` cells for new rows, then: python scripts/split_data.py")


if __name__ == "__main__":
    main()
