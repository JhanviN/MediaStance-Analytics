#!/usr/bin/env python3
"""
STEP 3 — Stratified train / test split from labeled_dataset.csv.

Requires every row to have label in {cooperative, neutral, adversarial}.
Drops rows with blank or invalid labels (prints counts).

Example:
  python scripts/split_data.py
  python scripts/split_data.py --test-size 0.25 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INPUT_DEFAULT = DATA / "labeled_dataset.csv"
TRAIN_OUT = DATA / "train.csv"
TEST_OUT = DATA / "test.csv"

VALID_LABELS = frozenset({"cooperative", "neutral", "adversarial"})


def main() -> None:
    ap = argparse.ArgumentParser(description="Stratified train/test split")
    ap.add_argument("-i", "--input", type=Path, default=INPUT_DEFAULT)
    ap.add_argument("--train-out", type=Path, default=TRAIN_OUT)
    ap.add_argument("--test-out", type=Path, default=TEST_OUT)
    ap.add_argument("--test-size", type=float, default=0.2, help="Fraction for test (0.2–0.3 typical)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"Missing {args.input}. Run: python scripts/init_labeled_template.py --latest")

    with open(args.input, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if "label" not in (rows[0].keys() if rows else []):
        sys.exit("Input CSV must include a 'label' column.")

    clean = []
    skipped_blank = 0
    skipped_bad = 0
    for r in rows:
        lab = (r.get("label") or "").strip().lower()
        if not lab:
            skipped_blank += 1
            continue
        if lab not in VALID_LABELS:
            skipped_bad += 1
            continue
        r = dict(r)
        r["label"] = lab
        clean.append(r)

    if skipped_blank or skipped_bad:
        print(f"Dropped {skipped_blank} blank-label rows, {skipped_bad} invalid-label rows.")

    if len(clean) < 10:
        sys.exit(f"Too few labeled rows ({len(clean)}). Need labels in {args.input}")

    # sklearn needs at least 2 samples per class for stratify
    labels = [r["label"] for r in clean]
    from collections import Counter

    cnt = Counter(labels)
    if any(v < 2 for v in cnt.values()):
        print(
            "Warning: some class has <2 examples; stratify may fail. "
            f"Class counts: {dict(cnt)}",
            file=sys.stderr,
        )
        try:
            train, test = train_test_split(
                clean,
                test_size=args.test_size,
                random_state=args.seed,
                stratify=labels,
            )
        except ValueError:
            train, test = train_test_split(
                clean, test_size=args.test_size, random_state=args.seed, stratify=None
            )
            print("Used random split (no stratify) due to rare classes.", file=sys.stderr)
    else:
        train, test = train_test_split(
            clean,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=labels,
        )

    fieldnames = list(clean[0].keys())
    for path, subset in [(args.train_out, train), (args.test_out, test)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in subset:
                w.writerow(r)

    print(f"Labeled rows used: {len(clean)}")
    print(f"Train: {len(train)} → {args.train_out}")
    print(f"Test:  {len(test)} → {args.test_out}")
    print("Train label counts:", dict(Counter(r["label"] for r in train)))
    print("Test label counts: ", dict(Counter(r["label"] for r in test)))


if __name__ == "__main__":
    main()
