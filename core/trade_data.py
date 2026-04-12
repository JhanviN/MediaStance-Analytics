"""
Load bilateral export-share snapshot and compute trade asymmetry.

Pair keys in JSON are always "{iso_a}-{iso_b}" with ISO codes sorted lexicographically.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

from . import config


def pair_key(a: str, b: str) -> str:
    x, y = sorted([a.upper(), b.upper()])
    return f"{x}-{y}"


def _snapshot_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    p = root / config.TRADE_SNAPSHOT_PATH
    if not p.exists():
        raise FileNotFoundError(f"Missing trade snapshot: {p}")
    return p


@lru_cache(maxsize=1)
def _load_snapshot() -> dict[str, Any]:
    with open(_snapshot_path(), encoding="utf-8") as f:
        return json.load(f)


def reload_trade_snapshot() -> None:
    _load_snapshot.cache_clear()


def _shares_for_ordered_pair(country_a: str, country_b: str, rec: dict) -> Tuple[float, float]:
    """Return (share_a_to_b, share_b_to_a) using iso_low/iso_high convention."""
    lo = rec["iso_low"].upper()
    hi = rec["iso_high"].upper()
    ca, cb = country_a.upper(), country_b.upper()
    if {ca, cb} != {lo, hi}:
        raise ValueError("Country mismatch vs snapshot record")

    s_lo_to_hi = float(rec["share_low_to_high"])
    s_hi_to_lo = float(rec["share_high_to_low"])

    if ca == lo and cb == hi:
        return s_lo_to_hi, s_hi_to_lo
    if ca == hi and cb == lo:
        return s_hi_to_lo, s_lo_to_hi
    raise ValueError("Unreachable pair orientation")


def get_trade_asymmetry(country_a: str, country_b: str) -> Dict:
    """
    Asymmetry = |share(A→B) − share(B→A)|, normalised 0–1 using dataset max (~0.20).
    leverage_holder: which country gains *supplier / export-side* structural advantage
    (higher export share to partner ⇒ partner depends more on those flows).
    """
    ca, cb = country_a.upper(), country_b.upper()
    key = pair_key(ca, cb)
    snap = _load_snapshot()
    block = snap.get("pairs", {}).get(key)

    if not block:
        return {
            "raw_asymmetry": 0.05,
            "normalized": 0.3,
            "note": "No bilateral snapshot for this pair — using conservative fallback",
            "share_a": None,
            "share_b": None,
            "leverage_holder": None,
            "pair_key": key,
        }

    share_a_to_b, share_b_to_a = _shares_for_ordered_pair(ca, cb, block)
    raw = abs(share_a_to_b - share_b_to_a)
    normalized = min(raw / 0.20, 1.0)

    if share_a_to_b > share_b_to_a:
        leverage_holder = ca
        asymmetry_note = (
            f"{ca} sends a larger share of its exports to {cb} than the reverse "
            f"— {cb} is relatively more dependent on {ca} as an export market"
        )
    elif share_b_to_a > share_a_to_b:
        leverage_holder = cb
        asymmetry_note = (
            f"{cb} sends a larger share of its exports to {ca} than the reverse "
            f"— {ca} is relatively more dependent on {cb} as an export market"
        )
    else:
        leverage_holder = "symmetric"
        asymmetry_note = "Near-symmetric export exposure between the two"

    return {
        "raw_asymmetry": round(raw, 4),
        "normalized": round(normalized, 4),
        "share_a": share_a_to_b,
        "share_b": share_b_to_a,
        "leverage_holder": leverage_holder,
        "note": asymmetry_note,
        "pair_key": key,
    }
