#!/usr/bin/env python3
"""
STEP 7 — Misclassification samples for the report.

Reads results/transformer_test_predictions.csv (or --pred baseline file),
writes results/error_analysis.md with up to N wrong examples.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred",
        type=Path,
        default=RESULTS / "transformer_test_predictions.csv",
    )
    ap.add_argument("--test-csv", type=Path, default=DATA / "test.csv")
    ap.add_argument("--out", type=Path, default=RESULTS / "error_analysis.md")
    ap.add_argument("--max-examples", type=int, default=25)
    args = ap.parse_args()

    if not args.pred.exists():
        raise SystemExit(f"Missing {args.pred} — run training first.")

    by_id = {}
    with open(args.test_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            by_id[row.get("id", "")] = row

    wrong = []
    with open(args.pred, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["true_label"].strip().lower() != row["pred_label"].strip().lower():
                wrong.append(row)

    lines = [
        "# Error analysis (test set)",
        "",
        f"Source predictions: `{args.pred.name}`",
        f"Total misclassified: **{len(wrong)}**",
        "",
    ]
    for i, row in enumerate(wrong[: args.max_examples]):
        meta = by_id.get(row.get("id", ""), {})
        head = meta.get("headline", "")[:200]
        pair = f"{meta.get('country_1','')}-{meta.get('country_2','')}"
        lines.append(f"### Example {i + 1}")
        lines.append("")
        lines.append(f"- **Pair:** {pair}")
        lines.append(f"- **Gold:** `{row['true_label']}`  **Pred:** `{row['pred_label']}`")
        lines.append(f"- **Headline:** {head}")
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out} ({min(len(wrong), args.max_examples)} examples shown)")


if __name__ == "__main__":
    main()
