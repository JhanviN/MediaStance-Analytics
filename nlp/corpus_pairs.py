"""
Target country pairs for bilateral corpus collection (Step 1).

Exactly four pairs — expand only after labels + baseline exist for this slice.
ISO2 codes; order in tuples is arbitrary (CSV stores sorted country_1, country_2).
"""

from __future__ import annotations

# High English RSS coverage + clear economic/geopolitical relevance
CORPUS_TARGET_PAIRS: list[tuple[str, str]] = [
    ("IN", "CN"),
    ("IN", "US"),
    ("CN", "US"),
    ("IN", "RU"),
]
