"""
TradePulse — Leverage Signal (submission core)

Two pillars only: trade asymmetry + news sentiment (with optional momentum vs last run).
Policy stance is out of scope until a labelled corpus exists — avoids a fake constant.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from . import config
from .trade_data import get_trade_asymmetry


def _sentiment_concern_0_1(mean_effective: float) -> float:
    """Map mean in [-1, 1] to concern in [0, 1]; high = more adversarial / tense."""
    return (1.0 - (mean_effective + 1.0) / 2.0)


def build_sentiment_state(
    mean: float,
    prior_mean: Optional[float],
    n: int,
    counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Blend current FinBERT-derived mean with momentum vs prior pipeline run."""
    delta: Optional[float] = None
    if prior_mean is not None:
        delta = round(mean - prior_mean, 4)

    if prior_mean is None:
        effective = mean
    else:
        effective = mean + config.SENTIMENT_MOMENTUM_COEFF * (mean - prior_mean)
        effective = max(-1.0, min(1.0, effective))

    concern = _sentiment_concern_0_1(effective)
    low_confidence = n < config.MIN_ARTICLES_FOR_FULL_CONFIDENCE

    out: Dict[str, Any] = {
        "mean": round(mean, 4),
        "prior_mean": None if prior_mean is None else round(prior_mean, 4),
        "delta": delta,
        "effective_mean": round(effective, 4),
        "concern_0_1": round(concern, 4),
        "n": n,
        "low_confidence": low_confidence,
    }
    if counts:
        out.update(counts)
    return out


def compute_leverage_score(
    country_a: str,
    country_b: str,
    trade_data: Dict,
    sentiment_state: Dict,
) -> Dict:
    """
    Leverage = WEIGHT_TRADE * trade_norm + WEIGHT_SENTIMENT * sentiment_concern
    scaled 0–10.
    """
    trade_score = float(trade_data["normalized"])
    sentiment_score = float(
        sentiment_state.get("concern_0_1")
        or _sentiment_concern_0_1(float(sentiment_state.get("effective_mean", sentiment_state.get("mean", 0.0))))
    )

    w_t = config.WEIGHT_TRADE
    w_s = config.WEIGHT_SENTIMENT
    leverage_raw = w_t * trade_score + w_s * sentiment_score
    leverage_10 = round(leverage_raw * 10, 2)

    if leverage_10 < 2.5:
        label, color = "Stable", "green"
    elif leverage_10 < 5.0:
        label, color = "Watchlist", "yellow"
    elif leverage_10 < 7.5:
        label, color = "Elevated", "orange"
    else:
        label, color = "Critical", "red"

    return {
        "pair": f"{country_a.upper()}↔{country_b.upper()}",
        "leverage_score": leverage_10,
        "label": label,
        "color": color,
        "components": {
            "trade_asymmetry": round(trade_score, 4),
            "sentiment_concern": round(sentiment_score, 4),
        },
        "weights": {"trade": w_t, "sentiment": w_s},
        "low_confidence": bool(sentiment_state.get("low_confidence")),
    }


def get_interpretation(
    score: float,
    pair: str,
    trade_data: Dict,
    sentiment_state: Dict,
) -> str:
    """Plain-English summary for reports / UI."""
    if score < 2.5:
        description = "relatively balanced"
    elif score < 5.0:
        description = "mildly asymmetric — worth monitoring"
    elif score < 7.5:
        description = "significantly imbalanced — structural or narrative pressure is visible"
    else:
        description = "critically imbalanced — strong leverage tension in the composite view"

    parts = [
        f"The {pair} relationship scores {score:.2f}/10 — {description}.",
        f"Trade structure: {trade_data.get('note', '')}",
    ]
    n = int(sentiment_state.get("n", 0))
    if sentiment_state.get("delta") is not None:
        parts.append(
            f"News tone (FinBERT, bilateral framing): mean {sentiment_state['mean']:+.2f} "
            f"vs prior {sentiment_state['prior_mean']:+.2f} (Δ {sentiment_state['delta']:+.2f}) over {n} articles."
        )
    else:
        parts.append(
            f"News tone (FinBERT, bilateral framing): mean {sentiment_state.get('mean', 0):+.2f} over {n} articles (no prior run for momentum)."
        )
    if sentiment_state.get("low_confidence"):
        parts.append("Low headline count for this pair in the window — treat sentiment as indicative, not definitive.")

    return " ".join(parts)


__all__ = [
    "get_trade_asymmetry",
    "build_sentiment_state",
    "compute_leverage_score",
    "get_interpretation",
]
