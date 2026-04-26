"""
Causality graph — built entirely from model predictions in predictions.db.

No CAMEO codes needed. Uses the model's own label + confidence + timestamp
to build event sequences and detect causal patterns.

Core idea:
  - Group predictions into 3-day windows per pair
  - Each window = a "state" (adversarial / cooperative / neutral)
  - Edges = state transitions over time
  - Spike analysis = what changed in the last N days vs prior N days
"""

from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from nlp.pair_utils import parse_pair, pair_key_from_codes

LABELS = ("adversarial", "cooperative", "neutral")
LABEL_COLORS = {
    "adversarial": "#F44336",
    "cooperative": "#4CAF50",
    "neutral":     "#2196F3",
}


# ── Core sequence query ───────────────────────────────────────────────────────

def prediction_sequence(
    conn: sqlite3.Connection,
    pair: str,
    *,
    days: int = 30,
    model: str = "baseline",
    min_confidence: float = 0.65,
) -> List[Dict[str, Any]]:
    """
    Return chronological list of high-confidence predictions for a pair.
    Each item: {date, label, confidence, headline}
    """
    a, b = parse_pair(pair)
    start = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()

    rows = conn.execute(
        """
        SELECT substr(created_at, 1, 10) as date,
               label, confidence, headline
        FROM predictions
        WHERE country_1 = ? AND country_2 = ?
          AND model = ?
          AND confidence >= ?
          AND substr(created_at, 1, 10) >= ?
        ORDER BY created_at ASC
        """,
        (a, b, model, min_confidence, start),
    ).fetchall()

    return [
        {
            "date": r["date"],
            "label": r["label"],
            "confidence": round(float(r["confidence"]), 3),
            "headline": (r["headline"] or "")[:120],
        }
        for r in rows
    ]


# ── Window-based state analysis ───────────────────────────────────────────────

def windowed_states(
    sequence: List[Dict[str, Any]],
    window_days: int = 3,
) -> List[Dict[str, Any]]:
    """
    Group predictions into N-day windows.
    Each window gets a dominant label (the most frequent label in that window).
    Returns list of {window_start, dominant_label, counts, headlines, confidence_avg}
    """
    if not sequence:
        return []

    # Find date range
    dates = sorted(set(ev["date"] for ev in sequence))
    if not dates:
        return []

    start_date = datetime.fromisoformat(dates[0])
    end_date = datetime.fromisoformat(dates[-1])

    windows = []
    current = start_date
    while current <= end_date:
        window_end = current + timedelta(days=window_days)
        window_start_str = current.date().isoformat()
        window_end_str = window_end.date().isoformat()

        # Collect predictions in this window
        window_preds = [
            ev for ev in sequence
            if window_start_str <= ev["date"] < window_end_str
        ]

        if window_preds:
            label_counts = Counter(ev["label"] for ev in window_preds)
            dominant = label_counts.most_common(1)[0][0]
            avg_conf = sum(ev["confidence"] for ev in window_preds) / len(window_preds)
            top_headlines = sorted(window_preds, key=lambda x: -x["confidence"])[:3]

            windows.append({
                "window_start": window_start_str,
                "window_end": window_end_str,
                "dominant_label": dominant,
                "counts": dict(label_counts),
                "total": len(window_preds),
                "avg_confidence": round(avg_conf, 3),
                "top_headlines": [h["headline"] for h in top_headlines],
            })

        current = window_end

    return windows


# ── Spike analysis ────────────────────────────────────────────────────────────

def spike_analysis(
    conn: sqlite3.Connection,
    pair: str,
    *,
    model: str = "baseline",
    window_days: int = 7,
    min_confidence: float = 0.65,
) -> Dict[str, Any]:
    """
    Compare label distribution in last window vs prior window.
    Returns what changed and the top headlines driving the change.
    """
    a, b = parse_pair(pair)
    now = datetime.now(timezone.utc).date()
    this_start = (now - timedelta(days=window_days)).isoformat()
    this_end = now.isoformat()
    prev_start = (now - timedelta(days=2 * window_days)).isoformat()
    prev_end = (now - timedelta(days=window_days)).isoformat()

    def _window_data(start: str, end: str) -> Dict[str, Any]:
        rows = conn.execute(
            """
            SELECT label, confidence, headline
            FROM predictions
            WHERE country_1 = ? AND country_2 = ?
              AND model = ?
              AND confidence >= ?
              AND substr(created_at, 1, 10) >= ?
              AND substr(created_at, 1, 10) < ?
            ORDER BY confidence DESC
            """,
            (a, b, model, min_confidence, start, end),
        ).fetchall()
        counts = Counter(r["label"] for r in rows)
        total = sum(counts.values())
        pct = {lab: round(100 * counts.get(lab, 0) / total, 1) if total else 0 for lab in LABELS}
        top = [{"headline": r["headline"][:120], "confidence": float(r["confidence"]), "label": r["label"]}
               for r in rows[:5]]
        return {"counts": dict(counts), "total": total, "pct": pct, "top_headlines": top}

    this = _window_data(this_start, this_end)
    prev = _window_data(prev_start, prev_end)

    # Compute deltas
    deltas = {
        lab: round(this["pct"].get(lab, 0) - prev["pct"].get(lab, 0), 1)
        for lab in LABELS
    }

    # Determine what's driving the change
    dominant_change = max(deltas, key=lambda x: abs(deltas[x]))
    direction = "increased" if deltas[dominant_change] > 0 else "decreased"

    # Build narrative
    if abs(deltas["adversarial"]) >= 5:
        if deltas["adversarial"] > 0:
            narrative = f"Adversarial sentiment rose by {deltas['adversarial']}pp — relationship is deteriorating."
        else:
            narrative = f"Adversarial sentiment fell by {abs(deltas['adversarial'])}pp — tensions easing."
    elif abs(deltas["cooperative"]) >= 5:
        if deltas["cooperative"] > 0:
            narrative = f"Cooperative sentiment rose by {deltas['cooperative']}pp — relationship improving."
        else:
            narrative = f"Cooperative sentiment fell by {abs(deltas['cooperative'])}pp — cooperation stalling."
    else:
        narrative = "No significant stance shift detected in this window."

    return {
        "pair": pair_key_from_codes(a, b),
        "window_days": window_days,
        "this_window": {**this, "start": this_start, "end": this_end},
        "prev_window": {**prev, "start": prev_start, "end": prev_end},
        "deltas": deltas,
        "narrative": narrative,
        "dominant_change": dominant_change,
    }


# ── Causal graph data ─────────────────────────────────────────────────────────

def build_causal_graph(
    conn: sqlite3.Connection,
    pair: str,
    *,
    days: int = 30,
    model: str = "baseline",
    window_days: int = 3,
    min_confidence: float = 0.65,
) -> Dict[str, Any]:
    """
    Build node/edge data for causality graph.

    Nodes = windowed states (each 3-day window is a node)
    Edges = transitions between consecutive windows
    Node size = number of predictions in that window
    Node color = dominant label
    Edge label = transition type (e.g., adversarial → cooperative)
    """
    sequence = prediction_sequence(
        conn, pair, days=days, model=model, min_confidence=min_confidence
    )

    if not sequence:
        return {
            "nodes": [], "edges": [], "pair": pair,
            "total_predictions": 0, "message": "No predictions found for this pair."
        }

    windows = windowed_states(sequence, window_days=window_days)

    if not windows:
        return {
            "nodes": [], "edges": [], "pair": pair,
            "total_predictions": len(sequence), "message": "Not enough data to build windows."
        }

    # Build nodes — one per window
    nodes = []
    for i, w in enumerate(windows):
        nodes.append({
            "id": f"w{i}",
            "label": w["dominant_label"],
            "window_start": w["window_start"],
            "window_end": w["window_end"],
            "total": w["total"],
            "counts": w["counts"],
            "avg_confidence": w["avg_confidence"],
            "top_headlines": w["top_headlines"],
            "color": LABEL_COLORS[w["dominant_label"]],
        })

    # Build edges — consecutive windows
    edges = []
    transition_counts: Counter = Counter()
    for i in range(len(windows) - 1):
        src_label = windows[i]["dominant_label"]
        tgt_label = windows[i + 1]["dominant_label"]
        transition_counts[(src_label, tgt_label)] += 1
        edges.append({
            "source": f"w{i}",
            "target": f"w{i+1}",
            "source_label": src_label,
            "target_label": tgt_label,
            "transition": f"{src_label} → {tgt_label}",
            "same": src_label == tgt_label,
        })

    # Transition summary
    transitions = [
        {"from": src, "to": tgt, "count": cnt, "transition": f"{src} → {tgt}"}
        for (src, tgt), cnt in transition_counts.most_common()
    ]

    # Find the most common path to adversarial
    adv_precursors = Counter()
    for i in range(1, len(windows)):
        if windows[i]["dominant_label"] == "adversarial":
            adv_precursors[windows[i-1]["dominant_label"]] += 1

    return {
        "pair": pair,
        "nodes": nodes,
        "edges": edges,
        "transitions": transitions,
        "adversarial_precursors": dict(adv_precursors),
        "total_predictions": len(sequence),
        "total_windows": len(windows),
        "date_range": f"{windows[0]['window_start']} → {windows[-1]['window_end']}",
    }


# ── Recurring patterns ────────────────────────────────────────────────────────

def recurring_patterns(
    conn: sqlite3.Connection,
    pair: str,
    *,
    days: int = 30,
    model: str = "baseline",
    window_days: int = 3,
    min_confidence: float = 0.65,
) -> List[Dict[str, Any]]:
    """
    Find recurring 3-step sequences (trigrams) in the state timeline.
    e.g., neutral → adversarial → adversarial appears 5 times
    """
    sequence = prediction_sequence(
        conn, pair, days=days, model=model, min_confidence=min_confidence
    )
    windows = windowed_states(sequence, window_days=window_days)
    labels = [w["dominant_label"] for w in windows]

    trigrams: Counter = Counter()
    for i in range(len(labels) - 2):
        trigram = (labels[i], labels[i+1], labels[i+2])
        trigrams[trigram] += 1

    return [
        {
            "pattern": f"{a} → {b} → {c}",
            "steps": [a, b, c],
            "count": cnt,
            "ends_adversarial": c == "adversarial",
        }
        for (a, b, c), cnt in trigrams.most_common(10)
        if cnt > 1
    ]
