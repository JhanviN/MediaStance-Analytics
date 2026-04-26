"""
Batch prediction — run a CSV of headlines through both models and save to predictions.db.

Usage:
    python scripts/predict_batch.py --input data/gdelt_raw.csv
    python scripts/predict_batch.py --input data/labeled_dataset_augmented.csv --model baseline
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.inference import predict_baseline, predict_transformer, combine_headline_body
from nlp.predictions_sqlite import connect, init_predictions_db, insert_prediction
from nlp.cameo_codes import get_description, normalize_code

VALID_LABELS = frozenset({"adversarial", "cooperative", "neutral"})


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch predict headlines → predictions.db")
    ap.add_argument("--input", type=Path, required=True, help="CSV with headline/text columns")
    ap.add_argument("--model", choices=["baseline", "transformer", "both"], default="both")
    ap.add_argument("--db", type=Path, default=ROOT / "data" / "predictions.db")
    ap.add_argument("--limit", type=int, default=0, help="Max rows to process (0=all)")
    ap.add_argument(
        "--skip-placeholders", action="store_true", default=True,
        help="Skip rows where headline starts with [GDELT (unenriched)"
    )
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"Input file not found: {args.input}")

    with open(args.input, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # Filter unenriched placeholders
    if args.skip_placeholders:
        before = len(rows)
        rows = [r for r in rows if not (r.get("headline") or "").startswith("[GDELT")]
        print(f"Filtered {before - len(rows)} placeholder rows. Remaining: {len(rows)}")

    if args.limit > 0:
        rows = rows[:args.limit]

    conn = connect(args.db)
    init_predictions_db(conn)

    processed = 0
    skipped = 0

    for i, row in enumerate(rows):
        headline = (row.get("headline") or "").strip()
        text = (row.get("text") or headline).strip()
        c1 = (row.get("country_1") or "").strip().upper()
        c2 = (row.get("country_2") or "").strip().upper()

        if not headline or len(headline) < 10:
            skipped += 1
            continue

        # Entity-aware input — same as training
        input_text = f"{c1}-{c2}: {text}" if c1 and c2 else text

        # Extract CAMEO code from source column (format: "GDELT-13")
        source = row.get("source", "")
        cameo_code = None
        cameo_desc = None
        if source.startswith("GDELT-"):
            raw_code = source.replace("GDELT-", "").strip()
            try:
                cameo_code = normalize_code(raw_code)
                cameo_desc = get_description(raw_code)
            except Exception:
                pass

        # Use published_at from source CSV as the prediction timestamp
        # This preserves the original article date for trend analysis
        published_at = (row.get("published_at") or "").strip() or None

        try:
            if args.model in ("baseline", "both"):
                lab, conf, probs = predict_baseline(input_text)
                insert_prediction(
                    conn,
                    headline=headline[:500],
                    text_used=input_text[:2000],
                    country_1=c1 or None,
                    country_2=c2 or None,
                    model="baseline",
                    label=lab,
                    confidence=conf,
                    probs=probs,
                    cameo_code=cameo_code,
                    cameo_description=cameo_desc,
                    created_at=published_at,
                )

            if args.model in ("transformer", "both"):
                lab, conf, probs = predict_transformer(input_text)
                insert_prediction(
                    conn,
                    headline=headline[:500],
                    text_used=input_text[:2000],
                    country_1=c1 or None,
                    country_2=c2 or None,
                    model="transformer",
                    label=lab,
                    confidence=conf,
                    probs=probs,
                    cameo_code=cameo_code,
                    cameo_description=cameo_desc,
                    created_at=published_at,
                )

            processed += 1
            if processed % 100 == 0:
                print(f"  [{processed}/{len(rows)}] processed...")

        except Exception as e:
            print(f"  [skip row {i}] {e}")
            skipped += 1
            continue

    conn.close()
    print(f"\nDone. Processed: {processed} | Skipped: {skipped}")
    print(f"Predictions saved → {args.db}")

    # Quick summary
    import sqlite3
    conn2 = sqlite3.connect(str(args.db))
    conn2.row_factory = sqlite3.Row
    total = conn2.execute("SELECT COUNT(*) as c FROM predictions").fetchone()["c"]
    pairs = conn2.execute(
        "SELECT country_1||'-'||country_2 as pair, COUNT(*) as c FROM predictions GROUP BY pair ORDER BY c DESC"
    ).fetchall()
    labels = conn2.execute(
        "SELECT label, COUNT(*) as c FROM predictions GROUP BY label"
    ).fetchall()
    conn2.close()

    print(f"\nTotal predictions in DB: {total}")
    print("By label:", {r["label"]: r["c"] for r in labels})
    print("By pair (top 5):", {r["pair"]: r["c"] for r in pairs[:5]})


if __name__ == "__main__":
    main()
