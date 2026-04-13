"""SQLite store for prediction CLI + API + weekly reports."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from nlp.pair_utils import normalize_stored_countries

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "predictions.db"


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_predictions_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            headline TEXT NOT NULL,
            text_used TEXT,
            country_1 TEXT,
            country_2 TEXT,
            model TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            p_adversarial REAL,
            p_cooperative REAL,
            p_neutral REAL,
            meta_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
        CREATE INDEX IF NOT EXISTS idx_predictions_label ON predictions(label);
        CREATE INDEX IF NOT EXISTS idx_predictions_pair ON predictions(country_1, country_2);
        CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model);
        """
    )
    conn.commit()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def insert_prediction(
    conn: sqlite3.Connection,
    *,
    headline: str,
    text_used: str,
    country_1: Optional[str],
    country_2: Optional[str],
    model: str,
    label: str,
    confidence: float,
    probs: Dict[str, float],
    meta: Optional[Dict[str, Any]] = None,
) -> int:
    c1, c2 = normalize_stored_countries(country_1, country_2)
    cur = conn.execute(
        """
        INSERT INTO predictions (
            created_at, headline, text_used, country_1, country_2,
            model, label, confidence, p_adversarial, p_cooperative, p_neutral, meta_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            utc_now_iso(),
            headline[:2000],
            text_used[:8000],
            c1,
            c2,
            model,
            label,
            confidence,
            probs.get("adversarial"),
            probs.get("cooperative"),
            probs.get("neutral"),
            json.dumps(meta or {}),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)
