
from __future__ import annotations

import hashlib
import html
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List, Tuple

import feedparser

from . import config

# ── Country alias map ─────────────────────────────────────────────────────────
COUNTRY_ALIASES = {
    "IN": ["india", "indian", "new delhi", "modi", "delhi", "rbi", "pib"],
    "CN": ["china", "chinese", "beijing", "xi jinping", "pboc", "shanghai"],
    "US": ["united states", "u.s.", "america", "american", "washington", "biden", "trump", "fed reserve", "ustr"],
    "RU": ["russia", "russian", "moscow", "putin", "kremlin"],
    "PK": ["pakistan", "pakistani", "islamabad", "lahore", "karachi", "imf pakistan"],
    "IR": ["iran", "iranian", "tehran", "khamenei", "irgc", "rouhani", "raisi"],
    "IL": ["israel", "israeli", "tel aviv", "jerusalem", "netanyahu", "idf", "gaza", "west bank"],
    # Kept for general RSS detection (not target pairs)
    "EU": ["european union", "europe", "brussels", "ecb", "eurozone", "european commission"],
    "JP": ["japan", "japanese", "tokyo", "bank of japan"],
    "DE": ["germany", "german", "berlin", "bundesbank"],
    "GB": ["uk", "britain", "british", "london", "bank of england"],
    "SA": ["saudi arabia", "saudi", "riyadh", "aramco"],
    "BR": ["brazil", "brazilian", "brasilia"],
    "AU": ["australia", "australian", "canberra"],
}

COUNTRY_NAMES = {
    "IN": "India",
    "CN": "China",
    "US": "USA",
    "RU": "Russia",
    "PK": "Pakistan",
    "IR": "Iran",
    "IL": "Israel",
    "EU": "EU",
    "JP": "Japan",
    "DE": "Germany",
    "GB": "UK",
    "SA": "Saudi Arabia",
    "BR": "Brazil",
    "AU": "Australia",
}


def _parse_published(entry: dict) -> datetime:
    raw = entry.get("published") or entry.get("updated") or ""
    if not raw:
        return datetime.now(timezone.utc)
    try:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    except Exception:
        pass
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def clean_rss_text(s: str) -> str:
    """Decode HTML entities, strip tags, normalize spaces (RSS/Google News snippets)."""
    if not s:
        return ""
    s = html.unescape(str(s))
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("\xa0", " ").replace("\u2009", " ").replace("\u2008", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_html(s: str) -> str:
    """Backward-compatible name for clean_rss_text."""
    return clean_rss_text(s)


def _word_jaccard(a: str, b: str) -> float:
    """Token overlap — catches '... - ThePrint' vs '... ThePrint' duplicates."""
    wa = set(re.findall(r"[\w']+", (a or "").lower()))
    wb = set(re.findall(r"[\w']+", (b or "").lower()))
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def build_article_text(title: str, summary: str) -> str:
    """
    Headline + optional extra context. RSS/Google often repeat the headline in the
    summary with only punctuation or outlet wording changed — collapse to one line.
    """
    title = clean_rss_text(title or "")
    summary = clean_rss_text(summary or "")
    if not summary:
        return title
    if summary.lower() == title.lower():
        return title
    jac = _word_jaccard(title, summary)
    # Same story twice (word sets ~ identical)
    if jac >= 0.91:
        return title
    if title and summary.lower().startswith(title.lower()) and len(summary) <= len(title) + 120:
        return title
    # Summary is mostly a repeat of the title (long near-duplicate)
    if jac >= 0.85 and len(summary) <= int(len(title) * 2.2) + 40:
        return title
    combined = f"{title}. {summary}"
    # "Title. Title ..." from feed quirks
    if title and combined.lower().count(title.lower()) >= 2 and len(summary) <= len(title) + 80:
        return title
    return combined


def _stable_url(entry: dict, title: str, text: str) -> str:
    link = (entry.get("link") or "").strip()
    if link:
        return link
    h = hashlib.sha256(f"{title}|{text}".encode("utf-8")).hexdigest()[:24]
    return f"urn:tradepulse:{h}"


def fetch_articles(max_per_feed: int = 20, max_age_days: int | None = None) -> List[Dict]:
    """Fetch recent headlines from configured RSS feeds."""
    articles: List[Dict] = []
    days = max_age_days if max_age_days is not None else config.MAX_ARTICLE_AGE_DAYS
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    for source, url in config.RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                title_raw = (entry.get("title") or "").strip()
                summary_raw = str(entry.get("summary", entry.get("description", "") or ""))
                title = clean_rss_text(title_raw) or title_raw
                text = build_article_text(title_raw, summary_raw)
                pub_dt = _parse_published(entry)
                if pub_dt < cutoff:
                    continue
                articles.append(
                    {
                        "source": source,
                        "title": title,
                        "text": text,
                        "url": _stable_url(entry, title, text),
                        "published": pub_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                )
        except Exception as e:
            print(f"  [Warning] Could not fetch {source}: {e}")
    return articles


def detect_countries(text: str) -> List[str]:
    text_lower = text.lower()
    found: List[str] = []
    for iso, aliases in COUNTRY_ALIASES.items():
        if any(alias in text_lower for alias in aliases):
            found.append(iso)
    return list(dict.fromkeys(found))


def filter_for_pair(articles: List[Dict], country_a: str, country_b: str) -> List[Dict]:
    relevant: List[Dict] = []
    for art in articles:
        countries = detect_countries(art["text"])
        if country_a in countries and country_b in countries:
            art["countries_detected"] = countries
            relevant.append(art)
    return relevant


if __name__ == "__main__":
    arts = fetch_articles()
    print(f"Total articles: {len(arts)}")
    pair = ("IN", "CN")
    rel = filter_for_pair(arts, *pair)
    print(f"{COUNTRY_NAMES[pair[0]]}↔{COUNTRY_NAMES[pair[1]]}: {len(rel)}")
    for a in rel[:5]:
        print(f"  [{a['source']}] {a['title'][:80]}")
