"""
Targeted RSS queries (live) — most wire headlines mention only one country.

These Google News search RSS URLs bias toward stories that discuss both sides
of a bilateral relationship. Titles are still vetted by labeling rules.
"""

from __future__ import annotations

from typing import List, TypedDict


class SupplementalFeed(TypedDict):
    pair: tuple[str, str]
    label: str
    url: str


# ceid biases locale; all are English RSS.
SUPPLEMENTAL_PAIR_FEEDS: List[SupplementalFeed] = [
    # ── Original 4 pairs ──────────────────────────────────────────────────────
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
    # ── New pairs ─────────────────────────────────────────────────────────────
    {
        "pair": ("IN", "PK"),
        "label": "GoogleNews RSS IN-PK",
        "url": "https://news.google.com/rss/search?q=India+Pakistan+trade+border+relations&hl=en&gl=IN&ceid=IN:en",
    },
    {
        "pair": ("IN", "IR"),
        "label": "GoogleNews RSS IN-IR",
        "url": "https://news.google.com/rss/search?q=India+Iran+oil+Chabahar+sanctions&hl=en&gl=IN&ceid=IN:en",
    },
    {
        "pair": ("IN", "IL"),
        "label": "GoogleNews RSS IN-IL",
        "url": "https://news.google.com/rss/search?q=India+Israel+defense+trade+relations&hl=en&gl=IN&ceid=IN:en",
    },
    {
        "pair": ("CN", "RU"),
        "label": "GoogleNews RSS CN-RU",
        "url": "https://news.google.com/rss/search?q=China+Russia+trade+energy+sanctions&hl=en&gl=US&ceid=US:en",
    },
    {
        "pair": ("CN", "IR"),
        "label": "GoogleNews RSS CN-IR",
        "url": "https://news.google.com/rss/search?q=China+Iran+oil+trade+BRI&hl=en&gl=US&ceid=US:en",
    },
    {
        "pair": ("CN", "PK"),
        "label": "GoogleNews RSS CN-PK",
        "url": "https://news.google.com/rss/search?q=China+Pakistan+CPEC+trade+economy&hl=en&gl=US&ceid=US:en",
    },
    {
        "pair": ("US", "RU"),
        "label": "GoogleNews RSS US-RU",
        "url": "https://news.google.com/rss/search?q=United+States+Russia+sanctions+Ukraine+economy&hl=en&gl=US&ceid=US:en",
    },
    {
        "pair": ("US", "IR"),
        "label": "GoogleNews RSS US-IR",
        "url": "https://news.google.com/rss/search?q=United+States+Iran+sanctions+nuclear+deal&hl=en&gl=US&ceid=US:en",
    },
    {
        "pair": ("US", "IL"),
        "label": "GoogleNews RSS US-IL",
        "url": "https://news.google.com/rss/search?q=United+States+Israel+military+aid+Gaza&hl=en&gl=US&ceid=US:en",
    },
    {
        "pair": ("IL", "IR"),
        "label": "GoogleNews RSS IL-IR",
        "url": "https://news.google.com/rss/search?q=Israel+Iran+conflict+nuclear+attack&hl=en&gl=US&ceid=US:en",
    },
    {
        "pair": ("RU", "IR"),
        "label": "GoogleNews RSS RU-IR",
        "url": "https://news.google.com/rss/search?q=Russia+Iran+drones+energy+sanctions&hl=en&gl=US&ceid=US:en",
    },
]
