"""
Targeted RSS queries (live) — most wire headlines mention only one country.

These Google News search RSS URLs bias toward stories that discuss both sides
of a bilateral economic relationship. Titles are still vetted by your labeling rules.
"""

from __future__ import annotations

from typing import List, TypedDict


class SupplementalFeed(TypedDict):
    pair: tuple[str, str]
    label: str
    url: str


# ceid=IN:en etc. biases locale; all are English RSS.
SUPPLEMENTAL_PAIR_FEEDS: List[SupplementalFeed] = [
    {
        "pair": ("IN", "CN"),
        "label": "GoogleNews RSS IN-CN",
        "url": "https://news.google.com/rss/search?q=India+China+trade+economy&hl=en&gl=IN&ceid=IN:en",
    },
    {
        "pair": ("IN", "US"),
        "label": "GoogleNews RSS IN-US",
        "url": "https://news.google.com/rss/search?q=India+United+States+trade+economy&hl=en&gl=IN&ceid=IN:en",
    },
    {
        "pair": ("CN", "US"),
        "label": "GoogleNews RSS CN-US",
        "url": "https://news.google.com/rss/search?q=China+United+States+trade+tariffs&hl=en&gl=US&ceid=US:en",
    },
    {
        "pair": ("IN", "RU"),
        "label": "GoogleNews RSS IN-RU",
        "url": "https://news.google.com/rss/search?q=India+Russia+trade+oil+economy&hl=en&gl=IN&ceid=IN:en",
    },
]
