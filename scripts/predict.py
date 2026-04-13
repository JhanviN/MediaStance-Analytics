#!/usr/bin/env python3
"""
Classify one headline/snippet with saved baseline and/or transformer.

  python scripts/predict.py -t "India and US hold trade talks amid tariff dispute"
  python scripts/predict.py -t "..." --model baseline --country1 IN --country2 US --save-db

--save-db appends a row to data/predictions.db (init schema on first use).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.inference import combine_headline_body, predict_baseline, predict_transformer  # noqa: E402
from nlp.pair_utils import pair_key_from_codes, parse_pair  # noqa: E402
from nlp.predictions_sqlite import connect, init_predictions_db, insert_prediction  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bilateral sentiment prediction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/predict.py -t "India and US hold trade talks"
  python scripts/predict.py -t "Headline" --body "Extra snippet" --model baseline
  python scripts/predict.py -t "..." --pair IN-US --save-db""",
    )
    ap.add_argument(
        "-t",
        "--text",
        default=None,
        metavar="TEXT",
        help="Headline or headline+snippet (same style as training `text` column); required",
    )
    ap.add_argument("--headline", default="", help="Optional short title for DB display (default: first 200 chars of --text)")
    ap.add_argument("--body", default="", help="Optional extra snippet combined with --headline if both set")
    ap.add_argument(
        "--pair",
        default="",
        help="Bilateral pair e.g. IN-US or US-CN (stored sorted in DB); overrides --country1/2",
    )
    ap.add_argument("--country1", default="", help="ISO2 e.g. IN (stored in DB)")
    ap.add_argument("--country2", default="", help="ISO2 e.g. US")
    ap.add_argument("--model", choices=["baseline", "transformer", "both"], default="both")
    ap.add_argument("--save-db", action="store_true", help="Append prediction(s) to data/predictions.db")
    ap.add_argument("--db", type=Path, default=None, help="SQLite path (default: data/predictions.db)")
    args = ap.parse_args()

    if args.text is None or not str(args.text).strip():
        print("predict.py: pass the text to classify with -t / --text", file=sys.stderr)
        print('  e.g. python scripts/predict.py -t "India and US trade talks"', file=sys.stderr)
        print("  Run with -h for full options.", file=sys.stderr)
        sys.exit(2)

    text = args.text.strip()
    if args.body.strip():
        text = combine_headline_body(text, args.body.strip())

    headline = (args.headline.strip() or text[:200]).strip()
    c1 = args.country1.strip().upper() or None
    c2 = args.country2.strip().upper() or None
    if args.pair.strip():
        try:
            c1, c2 = parse_pair(args.pair)
        except ValueError as e:
            print(f"predict.py: {e}", file=sys.stderr)
            sys.exit(2)

    out: dict = {"text_used": text}
    if c1 and c2:
        out["pair"] = pair_key_from_codes(c1, c2)

    if args.model in ("baseline", "both"):
        lab, conf, probs = predict_baseline(text)
        out["baseline"] = {"label": lab, "confidence": round(conf, 4), "probabilities": probs}
    if args.model in ("transformer", "both"):
        lab, conf, probs = predict_transformer(text)
        out["transformer"] = {"label": lab, "confidence": round(conf, 4), "probabilities": probs}

    print(json.dumps(out, indent=2))

    if args.save_db:
        conn = connect(args.db)
        init_predictions_db(conn)
        if "baseline" in out:
            b = out["baseline"]
            insert_prediction(
                conn,
                headline=headline,
                text_used=text,
                country_1=c1,
                country_2=c2,
                model="baseline",
                label=b["label"],
                confidence=b["confidence"],
                probs=b["probabilities"],
            )
        if "transformer" in out:
            t = out["transformer"]
            insert_prediction(
                conn,
                headline=headline,
                text_used=text,
                country_1=c1,
                country_2=c2,
                model="transformer",
                label=t["label"],
                confidence=t["confidence"],
                probs=t["probabilities"],
            )
        conn.close()
        print("(saved to SQLite)", file=sys.stderr)


if __name__ == "__main__":
    main()
