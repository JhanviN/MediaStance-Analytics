"""
TradePulse - Bilateral Sentiment Classifier
Uses FinBERT (ProsusAI) to score news articles as cooperative / neutral / adversarial
for a given country pair.

Zero-shot usage (before fine-tuning):
  python sentiment.py

After fine-tuning (Phase 2), swap MODEL_NAME to your HuggingFace Hub path.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
# Zero-shot: ProsusAI/finbert  (positive / negative / neutral → we remap)
# After fine-tuning: your own model path on HuggingFace Hub
MODEL_NAME = "ProsusAI/finbert"

# FinBERT label → bilateral sentiment mapping
# FinBERT positive  → cooperative  (trade deals, agreements, FTA progress)
# FinBERT negative  → adversarial  (tariffs, sanctions, disputes, bans)
# FinBERT neutral   → neutral
LABEL_MAP = {
    "positive": "cooperative",
    "negative": "adversarial",
    "neutral":  "neutral",
}

SCORE_MAP = {
    "cooperative": 1.0,
    "neutral":     0.0,
    "adversarial": -1.0,
}

_classifier = None  # lazy-loaded

def load_model():
    global _classifier
    if _classifier is None:
        print(f"Loading FinBERT model ({MODEL_NAME})...")
        _classifier = pipeline(
            "text-classification",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            return_all_scores=True,
            truncation=True,
            max_length=512,
        )
        print("Model loaded ✓")
    return _classifier


def classify_article(text: str, country_a: str, country_b: str) -> Dict:
    """
    Classify bilateral sentiment of a single article.

    Returns:
        label       : 'cooperative' | 'neutral' | 'adversarial'
        confidence  : float 0-1
        raw_scores  : dict of all label scores
        numeric     : -1.0 | 0.0 | 1.0
    """
    clf = load_model()
    # Prepend context so model understands bilateral framing
    context = f"Regarding the economic relationship between {country_a} and {country_b}: {text}"
    context = context[:1024]  # safety trim

    results = clf(context)[0]
    # results: [{"label": "positive", "score": 0.82}, ...]
    best = max(results, key=lambda x: x["score"])
    raw_label = best["label"].lower()
    mapped = LABEL_MAP.get(raw_label, "neutral")

    return {
        "label":      mapped,
        "confidence": round(best["score"], 4),
        "numeric":    SCORE_MAP[mapped],
        "raw_scores": {r["label"].lower(): round(r["score"], 4) for r in results},
    }


def score_articles(articles: List[Dict], country_a: str, country_b: str) -> List[Dict]:
    """Run classifier on a list of articles. Adds 'sentiment' key to each."""
    scored = []
    for art in articles:
        try:
            sentiment = classify_article(art["text"], country_a, country_b)
            scored.append({**art, "sentiment": sentiment})
        except Exception as e:
            print(f"  [Warning] Skipped article: {e}")
    return scored


def compute_pair_score(scored_articles: List[Dict]) -> Dict:
    """
    Aggregate article-level scores into a pair-level sentiment score.
    Returns mean numeric score + distribution.
    """
    if not scored_articles:
        return {"mean": 0.0, "label": "neutral", "n": 0,
                "cooperative": 0, "neutral": 0, "adversarial": 0}

    numerics   = [a["sentiment"]["numeric"] for a in scored_articles]
    labels     = [a["sentiment"]["label"]   for a in scored_articles]
    mean_score = sum(numerics) / len(numerics)

    # Majority label
    counts = {"cooperative": labels.count("cooperative"),
              "neutral":     labels.count("neutral"),
              "adversarial": labels.count("adversarial")}
    majority = max(counts, key=counts.get)

    return {
        "mean":        round(mean_score, 4),
        "label":       majority,
        "n":           len(scored_articles),
        **counts,
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_headlines = [
        "India imposes anti-dumping duty on Chinese steel imports, escalating trade tensions",
        "India and China agree to ease border tensions, resume normal trade operations",
        "India's trade deficit with China widens to record $85 billion in fiscal year",
        "China's PBOC cuts rates to stimulate exports amid India trade dispute concerns",
        "India and China hold diplomatic talks on bilateral trade normalisation",
    ]

    print("=" * 60)
    print("TradePulse Bilateral Sentiment — India ↔ China")
    print("=" * 60)

    articles = [{"text": h, "title": h, "source": "test"} for h in test_headlines]
    scored   = score_articles(articles, "India", "China")

    for art in scored:
        s = art["sentiment"]
        bar = {"cooperative": "🟢", "neutral": "🟡", "adversarial": "🔴"}[s["label"]]
        print(f"\n{bar} [{s['label'].upper():>13}] conf={s['confidence']:.2f}")
        print(f"   {art['title'][:72]}")

    pair_score = compute_pair_score(scored)
    print("\n" + "=" * 60)
    print(f"PAIR SCORE  →  {pair_score['mean']:+.3f}  ({pair_score['label'].upper()})")
    print(f"Articles    →  {pair_score['n']} total | "
          f"🟢 {pair_score['cooperative']} cooperative | "
          f"🟡 {pair_score['neutral']} neutral | "
          f"🔴 {pair_score['adversarial']} adversarial")
    print("=" * 60)
