"""Read-only analytics over ``predictions`` (SQLite). Uses ``model='baseline'`` by default to avoid double-counting."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

from nlp.corpus_pairs import CORPUS_TARGET_PAIRS
from nlp.pair_utils import pair_key_from_codes, parse_pair

LABELS = ("adversarial", "cooperative", "neutral")
DEFAULT_MODEL = "baseline"


def _pair_where() -> str:
    return "country_1 = ? AND country_2 = ?"


def _base_sql(model: str) -> str:
    return f"FROM predictions WHERE {_pair_where()} AND model = ?"


def label_distribution(
    conn: sqlite3.Connection,
    pair: str,
    *,
    model: str = DEFAULT_MODEL,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Percentages + counts for one canonical pair. Dates: YYYY-MM-DD inclusive (UTC string prefix match)."""
    a, b = parse_pair(pair)
    sql = f"SELECT label, COUNT(*) AS n {_base_sql(model)}"
    cond: List[Any] = [a, b, model]
    if start_date:
        sql += " AND substr(created_at, 1, 10) >= ?"
        cond.append(start_date[:10])
    if end_date:
        sql += " AND substr(created_at, 1, 10) <= ?"
        cond.append(end_date[:10])
    sql += " GROUP BY label"
    rows = conn.execute(sql, cond).fetchall()
    counts = {lab: 0 for lab in LABELS}
    for r in rows:
        lab = r["label"]
        if lab in counts:
            counts[lab] = int(r["n"])
    total = sum(counts.values())
    pct = {lab: (100.0 * counts[lab] / total) if total else 0.0 for lab in LABELS}
    return {
        "pair": pair_key_from_codes(a, b),
        "model": model,
        "total": total,
        "counts": counts,
        "percent": {k: round(v, 2) for k, v in pct.items()},
    }


def summary_all_pairs(
    conn: sqlite3.Connection,
    *,
    model: str = DEFAULT_MODEL,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for c1, c2 in CORPUS_TARGET_PAIRS:
        key = pair_key_from_codes(c1, c2)
        out[key] = label_distribution(conn, key, model=model, start_date=start_date, end_date=end_date)
    return {"pairs": out, "model": model}


def trends_by_day(
    conn: sqlite3.Connection,
    pair: str,
    *,
    model: str = DEFAULT_MODEL,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    a, b = parse_pair(pair)
    sql = (
        f"SELECT substr(created_at, 1, 10) AS d, label, COUNT(*) AS n "
        f"{_base_sql(model)}"
    )
    cond: List[Any] = [a, b, model]
    if start_date:
        sql += " AND substr(created_at, 1, 10) >= ?"
        cond.append(start_date[:10])
    if end_date:
        sql += " AND substr(created_at, 1, 10) <= ?"
        cond.append(end_date[:10])
    sql += " GROUP BY d, label ORDER BY d"
    rows = conn.execute(sql, cond).fetchall()
    by_day: Dict[str, Dict[str, int]] = defaultdict(lambda: {lab: 0 for lab in LABELS})
    for r in rows:
        d = r["d"]
        lab = r["label"]
        if lab in LABELS:
            by_day[d][lab] = int(r["n"])
    series: List[Dict[str, Any]] = []
    for d in sorted(by_day.keys()):
        c = by_day[d]
        tot = sum(c.values())
        if tot == 0:
            continue
        row: Dict[str, Any] = {
            "date": d,
            "cooperative": round(c["cooperative"] / tot, 4),
            "neutral": round(c["neutral"] / tot, 4),
            "adversarial": round(c["adversarial"] / tot, 4),
            "n": tot,
        }
        series.append(row)
    return series


def rolling_trends(
    series: List[Dict[str, Any]],
    window: int = 7,
    field: Literal["adversarial", "cooperative", "neutral"] = "adversarial",
) -> List[Dict[str, Any]]:
    """Simple trailing mean over ``field`` (by index order, not calendar gaps)."""
    if window < 1 or not series:
        return []
    vals = [float(s[field]) for s in series]
    out: List[Dict[str, Any]] = []
    for i in range(len(series)):
        lo = max(0, i - window + 1)
        chunk = vals[lo : i + 1]
        out.append(
            {
                "date": series[i]["date"],
                f"{field}_rolling_avg": round(sum(chunk) / len(chunk), 4),
                "n_points": len(chunk),
            }
        )
    return out


def list_headlines(
    conn: sqlite3.Connection,
    pair: str,
    *,
    model: str = DEFAULT_MODEL,
    label: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sort: Literal["date", "confidence"] = "date",
    limit: int = 100,
) -> List[Dict[str, Any]]:
    a, b = parse_pair(pair)
    sql = (
        "SELECT id, headline, label, confidence, created_at, model "
        "FROM predictions WHERE "
        + _pair_where()
        + " AND model = ?"
    )
    cond: List[Any] = [a, b, model]
    if label:
        sql += " AND label = ?"
        cond.append(label)
    if start_date:
        sql += " AND substr(created_at, 1, 10) >= ?"
        cond.append(start_date[:10])
    if end_date:
        sql += " AND substr(created_at, 1, 10) <= ?"
        cond.append(end_date[:10])
    sql += " ORDER BY " + ("confidence DESC" if sort == "confidence" else "created_at DESC")
    sql += f" LIMIT {max(1, min(int(limit), 500))}"
    rows = conn.execute(sql, cond).fetchall()
    return [
        {
            "id": r["id"],
            "headline": r["headline"],
            "label": r["label"],
            "confidence": float(r["confidence"]),
            "created_at": r["created_at"],
            "model": r["model"],
        }
        for r in rows
    ]


def pct_adversarial(dist: Dict[str, Any]) -> float:
    return float(dist["percent"]["adversarial"])


def detect_spikes(
    conn: sqlite3.Connection,
    *,
    model: str = DEFAULT_MODEL,
    threshold_pp: float = 15.0,
    days: int = 7,
) -> List[Dict[str, Any]]:
    """
    Compare last ``days`` vs previous ``days`` window (UTC calendar).
    If adversarial % rises by more than ``threshold_pp`` percentage points → alert.
    """
    now = datetime.now(timezone.utc).date()
    # Recent window: [now-7d, now] inclusive by date; prior: [now-14d, now-8d] so no overlap
    this_end = now.isoformat()
    this_start = (now - timedelta(days=days)).isoformat()
    prev_end = (now - timedelta(days=days + 1)).isoformat()
    prev_start = (now - timedelta(days=2 * days)).isoformat()

    alerts: List[Dict[str, Any]] = []
    for c1, c2 in CORPUS_TARGET_PAIRS:
        key = pair_key_from_codes(c1, c2)
        d_this = label_distribution(conn, key, model=model, start_date=this_start, end_date=this_end)
        d_prev = label_distribution(conn, key, model=model, start_date=prev_start, end_date=prev_end)
        if d_this["total"] < 3 and d_prev["total"] < 3:
            continue
        delta = pct_adversarial(d_this) - pct_adversarial(d_prev)
        if delta >= threshold_pp:
            sev = "high" if delta >= 25 else "medium"
            alerts.append(
                {
                    "pair": key,
                    "severity": sev,
                    "message": f"Adversarial share rose by about {delta:.1f} percentage points vs prior {days}d window (baseline rows only).",
                    "delta_adversarial_pp": round(delta, 2),
                    "window_days": days,
                    "this_window_n": d_this["total"],
                    "prev_window_n": d_prev["total"],
                }
            )
    return alerts


def compare_pairs(
    conn: sqlite3.Connection,
    pair1: str,
    pair2: str,
    *,
    model: str = DEFAULT_MODEL,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "pair1": label_distribution(conn, pair1, model=model, start_date=start_date, end_date=end_date),
        "pair2": label_distribution(conn, pair2, model=model, start_date=start_date, end_date=end_date),
        "model": model,
    }
