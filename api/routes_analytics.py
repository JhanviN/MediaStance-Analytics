

from __future__ import annotations

import sqlite3
from typing import Generator, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from nlp import analytics_queries as aq
from nlp.analytics_queries import DEFAULT_MODEL
from nlp.pair_utils import CANONICAL_PAIR_KEYS
from nlp.predictions_sqlite import connect, init_predictions_db

router = APIRouter(prefix="", tags=["analytics"])

_PAIR_EXAMPLES = ", ".join(sorted(CANONICAL_PAIR_KEYS))


def _require_pair(pair: Optional[str], *, name: str = "pair") -> str:
    if pair is None or not str(pair).strip():
        raise HTTPException(
            400,
            f"Missing query parameter '{name}'. Example: ?{name}=IN-US  "
            f"(any order is fine; stored keys: {_PAIR_EXAMPLES}).",
        )
    return str(pair).strip()


def get_db() -> Generator[sqlite3.Connection, None, None]:
    conn = connect()
    init_predictions_db(conn)
    try:
        yield conn
    finally:
        conn.close()


def _bad_pair(e: ValueError) -> HTTPException:
    return HTTPException(400, str(e))


@router.get("/distribution")
def get_distribution(
    pair: Optional[str] = Query(
        None,
        description="Bilateral pair (required), e.g. IN-US or US-CN",
        examples=["IN-US"],
    ),
    model: str = Query(DEFAULT_MODEL, description="Which model's rows to aggregate"),
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD inclusive"),
    end_date: Optional[str] = Query(None, description="YYYY-MM-DD inclusive"),
    conn: sqlite3.Connection = Depends(get_db),
):
    pair = _require_pair(pair)
    try:
        return aq.label_distribution(conn, pair, model=model, start_date=start_date, end_date=end_date)
    except ValueError as e:
        raise _bad_pair(e) from e


@router.get("/summary")
def get_summary(
    pair: Optional[str] = Query(None, description="Bilateral pair (required)", examples=["IN-US"]),
    model: str = Query(DEFAULT_MODEL),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    conn: sqlite3.Connection = Depends(get_db),
):
    """Same payload as ``/distribution`` (label shares for one pair)."""
    pair = _require_pair(pair)
    try:
        return aq.label_distribution(conn, pair, model=model, start_date=start_date, end_date=end_date)
    except ValueError as e:
        raise _bad_pair(e) from e


@router.get("/summary/all")
def get_summary_all(
    model: str = Query(DEFAULT_MODEL),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    conn: sqlite3.Connection = Depends(get_db),
):
    return aq.summary_all_pairs(conn, model=model, start_date=start_date, end_date=end_date)


@router.get("/trends")
def get_trends(
    pair: Optional[str] = Query(None, description="Bilateral pair (required)", examples=["IN-US"]),
    model: str = Query(DEFAULT_MODEL),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    rolling: int = Query(
        0,
        ge=0,
        le=30,
        description="Trailing window length for adversarial rolling average (0 = omit)",
    ),
    conn: sqlite3.Connection = Depends(get_db),
):
    pair = _require_pair(pair)
    try:
        series = aq.trends_by_day(conn, pair, model=model, start_date=start_date, end_date=end_date)
    except ValueError as e:
        raise _bad_pair(e) from e
    out: dict = {"pair": pair, "model": model, "series": series}
    if rolling > 0 and series:
        out["rolling_adversarial"] = aq.rolling_trends(series, window=rolling, field="adversarial")
    return out


@router.get("/headlines")
def get_headlines(
    pair: Optional[str] = Query(None, description="Bilateral pair (required)", examples=["IN-US"]),
    model: str = Query(DEFAULT_MODEL),
    label: Optional[str] = Query(None, description="adversarial | cooperative | neutral"),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sort: Literal["date", "confidence"] = "date",
    limit: int = Query(100, ge=1, le=500),
    conn: sqlite3.Connection = Depends(get_db),
):
    pair = _require_pair(pair)
    try:
        return {
            "pair": pair,
            "items": aq.list_headlines(
                conn,
                pair,
                model=model,
                label=label,
                start_date=start_date,
                end_date=end_date,
                sort=sort,
                limit=limit,
            ),
        }
    except ValueError as e:
        raise _bad_pair(e) from e


@router.get("/alerts")
def get_alerts(
    model: str = Query(DEFAULT_MODEL),
    threshold_pp: float = Query(15.0, ge=1.0, le=100.0, description="Min rise in adversarial % pts vs prior window"),
    days: int = Query(7, ge=1, le=60),
    conn: sqlite3.Connection = Depends(get_db),
):
    return {"alerts": aq.detect_spikes(conn, model=model, threshold_pp=threshold_pp, days=days)}


@router.get("/compare")
def get_compare(
    pair1: Optional[str] = Query(None, description="First pair (required)", examples=["IN-US"]),
    pair2: Optional[str] = Query(None, description="Second pair (required)", examples=["CN-US"]),
    model: str = Query(DEFAULT_MODEL),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    conn: sqlite3.Connection = Depends(get_db),
):
    pair1 = _require_pair(pair1, name="pair1")
    pair2 = _require_pair(pair2, name="pair2")
    try:
        return aq.compare_pairs(
            conn, pair1, pair2, model=model, start_date=start_date, end_date=end_date
        )
    except ValueError as e:
        raise _bad_pair(e) from e
