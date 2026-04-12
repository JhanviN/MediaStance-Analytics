"""
TradePulse — Streamlit demo (submission).

Run from repo root:
  streamlit run demo/demo_app.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path

import streamlit as st

from core import config
from core.leverage_engine import (
    build_sentiment_state,
    compute_leverage_score,
    get_interpretation,
    get_trade_asymmetry,
)
from core.news_fetcher import COUNTRY_NAMES, fetch_articles, filter_for_pair
from core.persistence import connect, init_db, latest_prior_mean
from core.sentiment import compute_pair_score, score_articles
from core.trade_data import pair_key

st.set_page_config(
    page_title="TradePulse — Economic Leverage Detector",
    page_icon="📡",
    layout="wide",
)

st.markdown(
    """
<h1 style='font-size:2.4rem;'>📡 TradePulse</h1>
<h3 style='color:#666; font-weight:400; margin-top:-10px;'>
Submission core — Trade asymmetry × News sentiment (FinBERT)</h3>
<hr/>
""",
    unsafe_allow_html=True,
)

db_path = str(Path(__file__).resolve().parents[1] / config.DEFAULT_DB_PATH)

with st.sidebar:
    st.header("Controls")
    SUPPORTED_PAIRS = list(config.SUBMISSION_PAIRS)
    pair_labels = [f"{COUNTRY_NAMES.get(a, a)} ↔ {COUNTRY_NAMES.get(b, b)}" for a, b in SUPPORTED_PAIRS]
    selected_idx = st.selectbox(
        "Country pair",
        range(len(pair_labels)),
        format_func=lambda i: pair_labels[i],
    )
    country_a, country_b = SUPPORTED_PAIRS[selected_idx]
    name_a, name_b = COUNTRY_NAMES.get(country_a, country_a), COUNTRY_NAMES.get(country_b, country_b)

    max_articles = st.slider("Max articles per feed", 5, 40, 20)
    run_btn = st.button("Analyse now", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown(
        f"""
**Submission scope**
- RSS corpus (last {config.MAX_ARTICLE_AGE_DAYS} days)
- Bilateral export shares: `data/trade_snapshot.json`
- **Weights:** {config.WEIGHT_TRADE:.0%} trade + {config.WEIGHT_SENTIMENT:.0%} sentiment  
- Policy stance: *deferred* (no neutral placeholder)
- Momentum: prior mean from `{config.DEFAULT_DB_PATH}`

**Repeatable run (CLI)**  
`python run_pipeline.py`
"""
    )

if not run_btn:
    st.info("Select a pair and click **Analyse now**.")
    st.markdown(
        """
### Workflow (what you submit)

1. **Ingest** — RSS headlines, deduped by URL, time-windowed.  
2. **Route** — keyword/alias filter → articles mentioning **both** countries.  
3. **NLP** — ProsusAI FinBERT with explicit bilateral framing prefix.  
4. **Structure** — export-share asymmetry from curated JSON (update yearly + cite source).  
5. **Composite** — weighted 0–10 score + label; **delta** vs last stored run when DB exists.  
6. **Persist** — SQLite enables momentum and audit trail for the report.

### Out of scope (add later)

GDELt, spaCy NER, stance classifier, Groq narratives, FastAPI/React cloud deploy.
"""
    )
    st.stop()

progress = st.progress(0, text="Starting…")

progress.progress(10, text=f"Fetching RSS for {name_a} ↔ {name_b}…")
with st.spinner("Fetching headlines…"):
    all_articles = fetch_articles(max_per_feed=max_articles)
    pair_articles = filter_for_pair(all_articles, country_a, country_b)

progress.progress(35, text=f"{len(all_articles)} articles fetched, {len(pair_articles)} for this pair")

conn = connect(db_path)
init_db(conn)
pkey = pair_key(country_a, country_b)
prior_mean = latest_prior_mean(conn, pkey)
conn.close()

trade_data = get_trade_asymmetry(country_a, country_b)
progress.progress(55, text="Trade asymmetry loaded")

if pair_articles:
    progress.progress(65, text="Running FinBERT…")
    with st.spinner("FinBERT inference…"):
        scored = score_articles(pair_articles, name_a, name_b)
    sentiment_dist = compute_pair_score(scored)
else:
    progress.progress(65, text="No live pair headlines — demo corpus")
    DEMO_HEADLINES = {
        ("IN", "CN"): [
            "India imposes anti-dumping duty on Chinese steel imports, escalating trade tensions",
            "India and China agree to ease border tensions, resume bilateral trade",
            "India's trade deficit with China hits record $85 billion this fiscal year",
            "India bans Chinese apps citing national security, tensions escalate",
            "India-China diplomatic talks on trade normalisation show progress",
        ],
        ("IN", "US"): [
            "US imposes tariffs on Indian steel and aluminium exports",
            "India and United States sign strategic trade partnership agreement",
            "India-US trade deal discussions advance, market access on agenda",
            "American companies boost investment in India amid supply chain shift",
            "India challenges US tariffs at WTO dispute settlement body",
        ],
        ("IN", "RU"): [
            "India continues to buy discounted Russian oil despite Western pressure",
            "India-Russia trade in rupee-ruble settlement faces banking hurdles",
            "Russia remains India's top oil supplier, bilateral trade surges",
            "India diversifies energy imports amid concerns over Russia dependency",
            "India and Russia sign long-term energy cooperation agreement",
        ],
    }
    key = (country_a, country_b) if (country_a, country_b) in DEMO_HEADLINES else ("IN", "CN")
    demo_arts = [{"text": h, "title": h, "source": "Demo corpus", "url": ""} for h in DEMO_HEADLINES[key]]
    with st.spinner("FinBERT on demo corpus…"):
        scored = score_articles(demo_arts, name_a, name_b)
    sentiment_dist = compute_pair_score(scored)
    st.warning("No headlines matched both countries in the RSS window — showing labelled demo lines for NLP.")

counts = {
    "cooperative": int(sentiment_dist.get("cooperative", 0)),
    "neutral": int(sentiment_dist.get("neutral", 0)),
    "adversarial": int(sentiment_dist.get("adversarial", 0)),
}
sentiment_state = build_sentiment_state(
    float(sentiment_dist["mean"]),
    prior_mean,
    int(sentiment_dist["n"]),
    counts,
)

progress.progress(85, text="Computing leverage…")
leverage = compute_leverage_score(country_a, country_b, trade_data, sentiment_state)
interpretation = get_interpretation(
    leverage["leverage_score"],
    f"{name_a}↔{name_b}",
    trade_data,
    sentiment_state,
)
progress.progress(100, text="Done")
progress.empty()

st.markdown(f"## {name_a} ↔ {name_b}")

c1, c2, c3, c4, c5 = st.columns(5)
score_color = {"Stable": "🟢", "Watchlist": "🟡", "Elevated": "🟠", "Critical": "🔴"}
with c1:
    st.metric("Leverage", f"{leverage['leverage_score']} / 10", delta=leverage["label"])
with c2:
    st.metric("Status", f"{score_color[leverage['label']]} {leverage['label']}")
with c3:
    d = sentiment_state.get("delta")
    st.metric("Sentiment mean", f"{sentiment_state['mean']:+.2f}", delta=None if d is None else f"Δ {d:+.2f}")
with c4:
    st.metric("Articles", str(sentiment_dist["n"]))
with c5:
    st.metric("Trade asym.", f"{trade_data['raw_asymmetry']:.1%}")

if leverage.get("low_confidence") or sentiment_state.get("low_confidence"):
    st.warning("Low article count for this window — sentiment is indicative only.")

bar_color = {"Stable": "#22c55e", "Watchlist": "#eab308", "Elevated": "#f97316", "Critical": "#ef4444"}[
    leverage["label"]
]
pct = leverage["leverage_score"] / 10
st.markdown(
    f"""
<div style='margin:12px 0;'>
  <div style='background:#e5e7eb; border-radius:8px; height:28px; width:100%;'>
    <div style='background:{bar_color}; border-radius:8px; height:28px; width:{pct*100:.0f}%;
                display:flex; align-items:center; padding-left:10px; color:white; font-weight:bold;'>
      {leverage['leverage_score']} / 10 — {leverage['label']}
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.info(f"**Analysis:** {interpretation}")

st.markdown("---")
st.markdown("### Components")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"#### Trade asymmetry ({leverage['weights']['trade']:.0%})")
    st.progress(leverage["components"]["trade_asymmetry"])
    st.caption(trade_data.get("note", ""))
    if trade_data.get("leverage_holder"):
        st.caption(f"Leverage holder (export exposure): **{trade_data['leverage_holder']}**")
with c2:
    st.markdown(f"#### Sentiment concern ({leverage['weights']['sentiment']:.0%})")
    st.progress(leverage["components"]["sentiment_concern"])
    st.caption(
        f"Effective mean (with momentum): **{sentiment_state['effective_mean']:+.3f}** "
        f"(concern 0–1: **{sentiment_state['concern_0_1']:.3f}**)"
    )

st.markdown("---")
st.markdown(f"### Articles ({len(scored)})")
label_emoji = {"cooperative": "🟢", "neutral": "🟡", "adversarial": "🔴"}
label_color = {"cooperative": "#dcfce7", "neutral": "#fef9c3", "adversarial": "#fee2e2"}
for art in scored[:20]:
    s = art["sentiment"]
    emoji = label_emoji[s["label"]]
    bg = label_color[s["label"]]
    with st.expander(f"{emoji} {art['title'][:90]}…  [{s['label'].upper()} | {s['confidence']:.2f}]"):
        st.markdown(f"**Source:** {art['source']}")
        st.markdown(f"**URL:** {art.get('url') or '—'}")
        for lbl, sc in s["raw_scores"].items():
            st.markdown(f"- {lbl}: `{sc:.3f}`")

st.markdown(
    """
---
<div style='text-align:center;color:#666;font-size:0.85rem;'>
TradePulse — submission core | SQLite + JSON trade snapshot | FinBERT
</div>
""",
    unsafe_allow_html=True,
)
