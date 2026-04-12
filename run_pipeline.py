#!/usr/bin/env python3
"""
CLI entry: run full TradePulse scoring pipeline and write reports/run_<id>.json

Usage (from repo root):
  python run_pipeline.py
  python run_pipeline.py --db data/tradepulse.db --feeds 25 --no-wb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="TradePulse submission pipeline")
    ap.add_argument("--db", default=None, help="SQLite path (default: data/tradepulse.db)")
    ap.add_argument("--feeds", type=int, default=20, help="Max articles per RSS feed")
    ap.add_argument("--no-wb", action="store_true", help="Skip World Bank macro lines")
    ap.add_argument("--notes", default="", help="Optional run label")
    args = ap.parse_args()

    summary = run_pipeline(
        db_path=args.db,
        max_per_feed=args.feeds,
        notes=args.notes,
        skip_worldbank=args.no_wb,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
