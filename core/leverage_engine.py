"""
TradePulse - Leverage Signal Engine
Combines trade asymmetry (structured) + news sentiment (NLP) into one score per pair.

For Phase 1 demo: trade asymmetry uses static World Bank snapshot data.
For Phase 2+: live World Bank API calls replace the static snapshot.
"""

from typing import Dict, Tuple

# ── Static trade asymmetry snapshot (World Bank 2022-23 data) ─────────────────
# ExportShare: what % of Country A's total exports go to Country B
# Source: World Bank WITS bilateral trade data
# Format: (ISO_A, ISO_B) → (share_A_to_B, share_B_to_A)
TRADE_SHARE_SNAPSHOT: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("IN", "CN"): (0.034, 0.032),   # India exports 3.4% to China; China exports 3.2% to India
    ("IN", "US"): (0.178, 0.021),   # India exports 17.8% to US; US exports 2.1% to India
    ("IN", "RU"): (0.021, 0.189),   # India exports 2.1% to Russia; Russia exports 18.9% to India (oil)
    ("IN", "EU"): (0.121, 0.018),   # India exports 12.1% to EU; EU exports 1.8% to India
    ("CN", "US"): (0.168, 0.085),   # China exports 16.8% to US; US exports 8.5% to China
    ("US", "EU"): (0.189, 0.201),   # US exports 18.9% to EU; EU exports 20.1% to US (near-symmetric)
    ("IN", "BD"): (0.034, 0.021),   # India-Bangladesh
    ("IN", "JP"): (0.029, 0.023),
    ("IN", "AU"): (0.043, 0.048),
    ("IN", "SA"): (0.031, 0.221),   # Saudi Arabia: energy leverage on India
}

# Normalise pair key (always smaller ISO first for lookup consistency)
def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a < b else (b, a)

def get_trade_asymmetry(country_a: str, country_b: str) -> Dict:
    """
    Compute trade asymmetry score for the pair.
    Asymmetry = |share(A→B) - share(B→A)|
    Normalised 0-1 against max observed asymmetry in dataset.

    Higher score = one country depends far more on the other = more leverage.
    """
    key = _pair_key(country_a, country_b)
    data = TRADE_SHARE_SNAPSHOT.get(key)

    if data is None:
        # Fallback: moderate unknown pair
        return {"raw_asymmetry": 0.05, "normalized": 0.3,
                "note": "No trade data — using fallback estimate",
                "share_a": None, "share_b": None}

    a_iso, b_iso = key
    share_a, share_b = data

    # If user asked in reverse order, swap labels
    if country_a == b_iso:
        share_a, share_b = share_b, share_a

    raw = abs(share_a - share_b)
    # Normalise: max asymmetry in our dataset ≈ 0.20 (Russia→India energy)
    normalized = min(raw / 0.20, 1.0)

    # Direction: who has the leverage?
    if share_a > share_b:
        leverage_holder = country_b   # B needs A more → A has leverage
        asymmetry_note = f"{country_a} exports more to {country_b}"
    elif share_b > share_a:
        leverage_holder = country_a
        asymmetry_note = f"{country_b} exports more to {country_a}"
    else:
        leverage_holder = "symmetric"
        asymmetry_note = "Near-symmetric trade relationship"

    return {
        "raw_asymmetry": round(raw, 4),
        "normalized":    round(normalized, 4),
        "share_a":       share_a,
        "share_b":       share_b,
        "leverage_holder": leverage_holder,
        "note": asymmetry_note,
    }


def compute_leverage_score(
    country_a: str,
    country_b: str,
    trade_data: Dict,
    sentiment_data: Dict,
    stance_data: Dict = None,
) -> Dict:
    """
    Leverage(A,B) = 0.40 × TradeAsymmetry + 0.35 × SentimentComponent + 0.25 × StanceDivergence
    → normalised to 0–10 scale

    For Phase 1 demo: StanceDivergence defaults to 0.5 (neutral/unknown).
    """
    # Component 1: Trade asymmetry (0-1)
    trade_score = trade_data["normalized"]  # already 0-1

    # Component 2: Sentiment (convert -1..+1 → 0..1, then invert:
    #   adversarial (-1) → high leverage concern (1.0)
    #   cooperative (+1) → low leverage concern (0.0)
    raw_sentiment = sentiment_data.get("mean", 0.0)
    sentiment_score = (1.0 - (raw_sentiment + 1.0) / 2.0)  # 0-1, high = adversarial

    # Component 3: Stance divergence (default 0.5 if not computed)
    stance_score = 0.5
    if stance_data:
        stance_score = stance_data.get("normalized", 0.5)

    # Weighted formula
    leverage_raw = (0.40 * trade_score) + (0.35 * sentiment_score) + (0.25 * stance_score)

    # Scale to 0-10
    leverage_10 = round(leverage_raw * 10, 2)

    # Label
    if leverage_10 < 2.5:
        label = "Stable"
        color = "green"
    elif leverage_10 < 5.0:
        label = "Watchlist"
        color = "yellow"
    elif leverage_10 < 7.5:
        label = "Elevated"
        color = "orange"
    else:
        label = "Critical"
        color = "red"

    return {
        "pair":             f"{country_a}↔{country_b}",
        "leverage_score":   leverage_10,
        "label":            label,
        "color":            color,
        "components": {
            "trade_asymmetry":    round(trade_score, 4),
            "sentiment":          round(sentiment_score, 4),
            "stance_divergence":  round(stance_score, 4),
        },
        "weights": {"trade": 0.40, "sentiment": 0.35, "stance": 0.25},
    }


def get_interpretation(score: float, pair: str, trade_data: Dict, sentiment_data: Dict) -> str:
    """Generate a plain-English interpretation of the leverage score."""
    label_map = {
        (0, 2.5): "relatively balanced",
        (2.5, 5): "mildly asymmetric with signs worth watching",
        (5, 7.5): "significantly imbalanced — one party holds meaningful economic coercive capacity",
        (7.5, 10): "critically imbalanced — strong economic leverage dynamic in play",
    }
    for (lo, hi), desc in label_map.items():
        if lo <= score < hi:
            description = desc
            break
    else:
        description = "in an extreme leverage imbalance"

    sentiment_label = sentiment_data.get("label", "neutral")
    trade_note = trade_data.get("note", "")

    return (
        f"The {pair} bilateral leverage relationship is {description}. "
        f"News sentiment is currently {sentiment_label} ({sentiment_data.get('n', 0)} articles analysed). "
        f"Trade structure: {trade_note}."
    )


if __name__ == "__main__":
    # Quick test with mock sentiment data
    trade = get_trade_asymmetry("IN", "CN")
    mock_sentiment = {"mean": -0.4, "label": "adversarial", "n": 8,
                      "cooperative": 1, "neutral": 2, "adversarial": 5}
    score = compute_leverage_score("IN", "CN", trade, mock_sentiment)

    print("=" * 50)
    print(f"LEVERAGE SCORE: {score['leverage_score']} / 10  [{score['label'].upper()}]")
    print(f"  Trade Asymmetry : {score['components']['trade_asymmetry']:.3f}")
    print(f"  Sentiment       : {score['components']['sentiment']:.3f}")
    print(f"  Stance          : {score['components']['stance_divergence']:.3f}")
    print()
    print(get_interpretation(score['leverage_score'], "IN↔CN", trade, mock_sentiment))
