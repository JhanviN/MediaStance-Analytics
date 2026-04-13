#!/usr/bin/env python3
"""
Aggregate predictions.db for the last 7 days vs the previous 7 days.

Requires rows written by: python scripts/predict.py ... --save-db
(or your own inserts into the same schema).

Output: results/weekly_report_YYYY-MM-DD.txt
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.predictions_sqlite import init_predictions_db  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _counts(conn: sqlite3.Connection, start: str, end: str | None) -> Counter:
    if end:
        rows = conn.execute(
            "SELECT label FROM predictions WHERE created_at >= ? AND created_at < ?",
            (start, end),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT label FROM predictions WHERE created_at >= ?",
            (start,),
        ).fetchall()
    return Counter(r["label"] for r in rows)


def _top_adversarial(conn: sqlite3.Connection, start: str, limit: int) -> list:
    return list(
        conn.execute(
            """
            SELECT headline, confidence, model, created_at
            FROM predictions
            WHERE created_at >= ? AND label = 'adversarial'
            ORDER BY confidence DESC
            LIMIT ?
            """,
            (start, limit),
        ).fetchall()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DATA / "predictions.db")
    ap.add_argument("--days", type=int, default=7, help="Length of each window")
    ap.add_argument("--top", type=int, default=10, help="Top adversarial headlines to list")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}. Run predict.py with --save-db first.")
        return

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    init_predictions_db(conn)
    n = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]
    if n == 0:
        print(f"Database empty: {args.db}")
        conn.close()
        return

    now = datetime.now(timezone.utc)
    this_start = now - timedelta(days=args.days)
    prev_start = now - timedelta(days=2 * args.days)
    this_s, prev_s = _iso(this_start), _iso(prev_start)
    this_e = _iso(now)

    this_cnt = _counts(conn, this_s, this_e)
    prev_cnt = _counts(conn, prev_s, this_s)
    total_this = sum(this_cnt.values())
    total_prev = sum(prev_cnt.values())

    lines = [
        f"Weekly report (UTC) generated {_iso(now)}",
        f"Window A (last {args.days}d): {this_s} .. {this_e}",
        f"Window B (prior {args.days}d): {prev_s} .. {this_s}",
        "",
        f"Predictions in A: {total_this} | in B: {total_prev}",
        "",
        "## Label distribution (window A)",
    ]
    for lab in ["adversarial", "cooperative", "neutral"]:
        c = this_cnt.get(lab, 0)
        pct = (100.0 * c / total_this) if total_this else 0.0
        lines.append(f"  {lab}: {c} ({pct:.1f}%)")

    lines.extend(["", "## Delta vs window B (count A - count B)"])
    for lab in ["adversarial", "cooperative", "neutral"]:
        d = this_cnt.get(lab, 0) - prev_cnt.get(lab, 0)
        lines.append(f"  {lab}: {d:+}")

    lines.extend(["", f"## Top {args.top} adversarial (window A, by confidence)"])
    tops = _top_adversarial(conn, this_s, args.top)
    if not tops:
        lines.append("  (none)")
    else:
        for r in tops:
            h = (r["headline"] or "")[:120].replace("\n", " ")
            lines.append(f"  - [{r['model']}] conf={r['confidence']:.3f} | {h}")

    conn.close()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"weekly_report_{now.date().isoformat()}.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
