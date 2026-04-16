

from __future__ import annotations

# Country pairs (ISO2). Kept aligned with `nlp.corpus_pairs.CORPUS_TARGET_PAIRS`.
SUBMISSION_PAIRS: list[tuple[str, str]] = [
    ("IN", "CN"), ("IN", "US"), ("CN", "US"), ("IN", "RU"),
    ("IN", "PK"), ("IN", "IR"), ("IN", "IL"),
    ("CN", "RU"), ("CN", "IR"), ("CN", "PK"),
    ("US", "RU"), ("US", "IR"), ("US", "IL"),
    ("IL", "IR"), ("RU", "IR"),
]

# Two-pillar composite (policy stance deferred — avoids neutral stance constant).
WEIGHT_TRADE = 0.55
WEIGHT_SENTIMENT = 0.45

# Sentiment: blend current mean with momentum vs last stored run.
SENTIMENT_MOMENTUM_COEFF = 0.45

# Low article count → flag in UI / reports
MIN_ARTICLES_FOR_FULL_CONFIDENCE = 5

# RSS only for submission core (GDELT = future work).
RSS_FEEDS: dict[str, str] = {
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
    "Reuters World": "https://feeds.reuters.com/reuters/worldNews",
    "PIB India": "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3",
    "BBC Business": "http://feeds.bbci.co.uk/news/business/rss.xml",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "Economic Times": "https://economictimes.indiatimes.com/rssfeedstopstories.cms",
}

DEFAULT_DB_PATH = "data/tradepulse.db"
TRADE_SNAPSHOT_PATH = "data/trade_snapshot.json"

# Corpus / RSS: drop items older than this (aligned to ~last 3 months for all pairs)
MAX_ARTICLE_AGE_DAYS = 90
