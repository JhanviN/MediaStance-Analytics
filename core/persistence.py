"""
SQLite persistence for repeatable submission workflow:
ingested articles + per-run pair scores (enables sentiment delta).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import config


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(db_path: str | Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            url TEXT NOT NULL,
            source TEXT,
            title TEXT,
            body TEXT,
            published_at TEXT,
            UNIQUE(url),
            FOREIGN KEY(run_id) REFERENCES pipeline_runs(id)
        );

        CREATE TABLE IF NOT EXISTS pair_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            pair_key TEXT NOT NULL,
            country_a TEXT NOT NULL,
            country_b TEXT NOT NULL,
            n_articles INTEGER NOT NULL,
            mean_sentiment REAL,
            prior_mean_sentiment REAL,
            sentiment_delta REAL,
            trade_normalized REAL,
            leverage_score REAL,
            label TEXT,
            interpretation TEXT,
            low_confidence INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY(run_id) REFERENCES pipeline_runs(id)
        );

        CREATE INDEX IF NOT EXISTS idx_pair_scores_pair ON pair_scores(pair_key);
        CREATE INDEX IF NOT EXISTS idx_pair_scores_run ON pair_scores(run_id);
        """
    )
    conn.commit()


def start_run(conn: sqlite3.Connection, notes: str = "") -> int:
    cur = conn.execute(
        "INSERT INTO pipeline_runs (started_at, notes) VALUES (?, ?)",
        (_utc_now(), notes),
    )
    conn.commit()
    return int(cur.lastrowid)


def finish_run(conn: sqlite3.Connection, run_id: int) -> None:
    conn.execute(
        "UPDATE pipeline_runs SET finished_at = ? WHERE id = ?",
        (_utc_now(), run_id),
    )
    conn.commit()


def upsert_article(
    conn: sqlite3.Connection,
    run_id: int,
    url: str,
    source: str,
    title: str,
    body: str,
    published_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO articles (run_id, url, source, title, body, published_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            run_id = excluded.run_id,
            source = excluded.source,
            title = excluded.title,
            body = excluded.body,
            published_at = excluded.published_at
        """,
        (run_id, url, source, title, body, published_at),
    )


def latest_prior_mean(conn: sqlite3.Connection, pair_key: str) -> Optional[float]:
    row = conn.execute(
        """
        SELECT mean_sentiment FROM pair_scores
        WHERE pair_key = ? AND mean_sentiment IS NOT NULL
        ORDER BY id DESC LIMIT 1
        """,
        (pair_key,),
    ).fetchone()
    if row and row["mean_sentiment"] is not None:
        return float(row["mean_sentiment"])
    return None


def insert_pair_result(
    conn: sqlite3.Connection,
    run_id: int,
    pair_key: str,
    country_a: str,
    country_b: str,
    payload: Dict[str, Any],
) -> None:
    s = payload["sentiment_state"]
    t = payload["trade_data"]
    lev = payload["leverage"]
    conn.execute(
        """
        INSERT INTO pair_scores (
            run_id, pair_key, country_a, country_b,
            n_articles, mean_sentiment, prior_mean_sentiment, sentiment_delta,
            trade_normalized, leverage_score, label, interpretation, low_confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            pair_key,
            country_a,
            country_b,
            int(s["n"]),
            s.get("mean"),
            s.get("prior_mean"),
            s.get("delta"),
            float(t["normalized"]),
            float(lev["leverage_score"]),
            lev["label"],
            payload.get("interpretation", ""),
            1 if lev.get("low_confidence") or s.get("low_confidence") else 0,
        ),
    )
    conn.commit()


def fetch_last_run_summary(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    run_id_row = conn.execute(
        "SELECT id FROM pipeline_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if not run_id_row:
        return []
    rid = int(run_id_row["id"])
    return list(
        conn.execute(
            "SELECT * FROM pair_scores WHERE run_id = ? ORDER BY leverage_score DESC",
            (rid,),
        )
    )
