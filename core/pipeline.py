"""
End-to-end submission pipeline:
  fetch RSS → filter pairs → FinBERT → trade asymmetry → leverage → SQLite
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from . import config, persistence
from .leverage_engine import (
    build_sentiment_state,
    compute_leverage_score,
    get_interpretation,
    get_trade_asymmetry,
)
from .news_fetcher import COUNTRY_NAMES, fetch_articles, filter_for_pair
from .sentiment import compute_pair_score, score_articles
from .trade_data import pair_key
from .worldbank import macro_context_line


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_pipeline(
    db_path: str | None = None,
    max_per_feed: int = 20,
    notes: str = "",
    pairs: List[Tuple[str, str]] | None = None,
    skip_worldbank: bool = False,
) -> Dict[str, Any]:
    """Execute one full scoring run. Returns JSON-serialisable summary."""
    db = db_path or str(_repo_root() / config.DEFAULT_DB_PATH)
    conn = persistence.connect(db)
    persistence.init_db(conn)
    run_id = persistence.start_run(conn, notes=notes)

    pairs = pairs or config.SUBMISSION_PAIRS
    all_articles = fetch_articles(max_per_feed=max_per_feed)

    summary_pairs: List[Dict[str, Any]] = []
    macro_lines: List[str] = []

    if not skip_worldbank:
        seen_iso: set[str] = set()
        for a, b in pairs:
            seen_iso.add(a.upper())
            seen_iso.add(b.upper())
        for iso in sorted(seen_iso):
            if iso == "EU":
                continue
            try:
                macro_lines.append(macro_context_line(iso))
            except Exception:
                pass

    for country_a, country_b in pairs:
        ca, cb = country_a.upper(), country_b.upper()
        pkey = pair_key(ca, cb)
        name_a = COUNTRY_NAMES.get(ca, ca)
        name_b = COUNTRY_NAMES.get(cb, cb)

        pair_articles = filter_for_pair(all_articles, ca, cb)
        for art in pair_articles:
            persistence.upsert_article(
                conn,
                run_id,
                art.get("url") or f"urn:tradepulse:{hash(art.get('text',''))}",
                art.get("source", ""),
                art.get("title", ""),
                art.get("text", ""),
                art.get("published", ""),
            )

        if pair_articles:
            scored = score_articles(pair_articles, name_a, name_b)
            dist = compute_pair_score(scored)
        else:
            scored, dist = [], {
                "mean": 0.0,
                "label": "neutral",
                "n": 0,
                "cooperative": 0,
                "neutral": 0,
                "adversarial": 0,
            }

        prior = persistence.latest_prior_mean(conn, pkey)
        sentiment_state = build_sentiment_state(
            float(dist["mean"]),
            prior,
            int(dist["n"]),
            {
                "cooperative": int(dist.get("cooperative", 0)),
                "neutral": int(dist.get("neutral", 0)),
                "adversarial": int(dist.get("adversarial", 0)),
            },
        )
        trade_data = get_trade_asymmetry(ca, cb)
        lev = compute_leverage_score(ca, cb, trade_data, sentiment_state)
        interp = get_interpretation(
            lev["leverage_score"],
            f"{name_a}↔{name_b}",
            trade_data,
            sentiment_state,
        )

        persistence.insert_pair_result(
            conn,
            run_id,
            pkey,
            ca,
            cb,
            {
                "sentiment_state": sentiment_state,
                "trade_data": trade_data,
                "leverage": lev,
                "interpretation": interp,
            },
        )

        summary_pairs.append(
            {
                "pair": f"{ca}-{cb}",
                "leverage_score": lev["leverage_score"],
                "label": lev["label"],
                "n_articles": dist["n"],
                "mean_sentiment": sentiment_state["mean"],
                "sentiment_delta": sentiment_state.get("delta"),
                "low_confidence": lev.get("low_confidence")
                or sentiment_state.get("low_confidence"),
            }
        )

    persistence.finish_run(conn, run_id)
    conn.close()

    out = {
        "run_id": run_id,
        "db_path": db,
        "pairs_scored": len(summary_pairs),
        "results": summary_pairs,
        "macro_context": macro_lines,
    }
    reports_dir = _repo_root() / "reports"
    reports_dir.mkdir(exist_ok=True)
    rep_path = reports_dir / f"run_{run_id}.json"
    rep_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    out["report_path"] = str(rep_path)
    return out
