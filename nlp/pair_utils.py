"""Canonical bilateral pair strings for API + SQLite filters."""

from __future__ import annotations

from typing import Optional

from nlp.corpus_pairs import CORPUS_TARGET_PAIRS

# Sorted "XX-YY" keys matching how we store country_1, country_2 after normalization
CANONICAL_PAIR_KEYS: frozenset[str] = frozenset(
    f"{a}-{b}" for a, b in (tuple(sorted(p)) for p in CORPUS_TARGET_PAIRS)
)


def sorted_iso2(c1: str, c2: str) -> tuple[str, str]:
    a, b = c1.strip().upper(), c2.strip().upper()
    return tuple(sorted((a, b)))


def pair_key_from_codes(c1: str, c2: str) -> str:
    a, b = sorted_iso2(c1, c2)
    return f"{a}-{b}"


def parse_pair(pair: str) -> tuple[str, str]:
    """
    Accept 'IN-US', 'US-IN', 'US–CN' (en dash) → sorted ISO2 codes.
    Raises ValueError if unknown or malformed.
    """
    if not pair or not str(pair).strip():
        raise ValueError("pair is empty")
    s = str(pair).strip().upper().replace("–", "-").replace("—", "-")
    parts = [p for p in s.split("-") if p]
    if len(parts) != 2:
        raise ValueError(f"expected XX-YY, got {pair!r}")
    a, b = sorted((parts[0], parts[1]))
    key = f"{a}-{b}"
    if key not in CANONICAL_PAIR_KEYS:
        raise ValueError(f"unknown pair {pair!r}; allowed: {sorted(CANONICAL_PAIR_KEYS)}")
    return a, b


def normalize_stored_countries(
    country_1: Optional[str], country_2: Optional[str]
) -> tuple[str, str]:
    """For DB rows: sort both ISO2 when both present; otherwise keep as given (trimmed)."""
    a = (country_1 or "").strip().upper()
    b = (country_2 or "").strip().upper()
    if a and b:
        return tuple(sorted((a, b)))
    return a, b
