"""
TradePulse - News Fetcher
Pulls real-time headlines from RSS feeds and filters for country pair mentions.
"""

import feedparser
import requests
from datetime import datetime, timezone
from typing import List, Dict, Tuple

# ── Country alias map ─────────────────────────────────────────────────────────
COUNTRY_ALIASES = {
    "IN": ["india", "indian", "new delhi", "modi", "delhi", "rbi", "pib"],
    "CN": ["china", "chinese", "beijing", "xi jinping", "pboc", "shanghai"],
    "US": ["united states", "america", "american", "washington", "biden", "trump", "fed reserve", "ustr"],
    "RU": ["russia", "russian", "moscow", "putin", "kremlin"],
    "EU": ["european union", "europe", "brussels", "ecb", "eurozone"],
    "JP": ["japan", "japanese", "tokyo", "bank of japan"],
    "DE": ["germany", "german", "berlin", "bundesbank"],
    "GB": ["uk", "britain", "british", "london", "bank of england"],
    "SA": ["saudi arabia", "saudi", "riyadh", "aramco"],
    "BR": ["brazil", "brazilian", "brasilia"],
    "BD": ["bangladesh", "dhaka"],
    "AU": ["australia", "australian", "canberra"],
}

COUNTRY_NAMES = {
    "IN": "India", "CN": "China", "US": "USA", "RU": "Russia",
    "EU": "EU", "JP": "Japan", "DE": "Germany", "GB": "UK",
    "SA": "Saudi Arabia", "BR": "Brazil", "BD": "Bangladesh", "AU": "Australia",
}

# ── RSS Feed sources ──────────────────────────────────────────────────────────
RSS_FEEDS = {
    "Reuters Business":   "https://feeds.reuters.com/reuters/businessNews",
    "Reuters World":      "https://feeds.reuters.com/reuters/worldNews",
    "PIB India":          "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3",
    "BBC Business":       "http://feeds.bbci.co.uk/news/business/rss.xml",
    "Al Jazeera Economy": "https://www.aljazeera.com/xml/rss/all.xml",
    "Economic Times":     "https://economictimes.indiatimes.com/rssfeedstopstories.cms",
}

def fetch_articles(max_per_feed: int = 20) -> List[Dict]:
    """Fetch articles from all RSS feeds."""
    articles = []
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                title = entry.get("title", "").strip()
                summary = entry.get("summary", entry.get("description", "")).strip()
                # Clean HTML tags simply
                import re
                summary = re.sub(r"<[^>]+>", " ", summary).strip()
                text = f"{title}. {summary}"
                articles.append({
                    "source": source,
                    "title": title,
                    "text": text,
                    "url": entry.get("link", ""),
                    "published": entry.get("published", str(datetime.now())),
                })
        except Exception as e:
            print(f"  [Warning] Could not fetch {source}: {e}")
    return articles


def detect_countries(text: str) -> List[str]:
    """Return list of ISO codes mentioned in text."""
    text_lower = text.lower()
    found = []
    for iso, aliases in COUNTRY_ALIASES.items():
        if any(alias in text_lower for alias in aliases):
            found.append(iso)
    return list(set(found))


def filter_for_pair(articles: List[Dict], country_a: str, country_b: str) -> List[Dict]:
    """Keep only articles mentioning both countries in a pair."""
    relevant = []
    for art in articles:
        countries = detect_countries(art["text"])
        if country_a in countries and country_b in countries:
            art["countries_detected"] = countries
            relevant.append(art)
    return relevant


if __name__ == "__main__":
    print("Fetching live news...")
    arts = fetch_articles()
    print(f"Total articles fetched: {len(arts)}")

    pair = ("IN", "CN")
    filtered = filter_for_pair(arts, *pair)
    print(f"\nArticles for {COUNTRY_NAMES[pair[0]]} ↔ {COUNTRY_NAMES[pair[1]]}: {len(filtered)}")
    for a in filtered[:5]:
        print(f"  [{a['source']}] {a['title'][:80]}")
