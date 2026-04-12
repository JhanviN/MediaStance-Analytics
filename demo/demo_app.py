"""
TradePulse - Live Demo App
Run with: streamlit run demo_app.py

Shows:
  1. Live news fetch for any country pair
  2. FinBERT bilateral sentiment analysis
  3. Leverage Score computation
  4. Article-level breakdown

This is the Phase 1+2 demo submission.
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.news_fetcher   import fetch_articles, filter_for_pair, COUNTRY_NAMES
from core.sentiment      import score_articles, compute_pair_score
from core.leverage_engine import get_trade_asymmetry, compute_leverage_score, get_interpretation

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TradePulse — Economic Leverage Detector",
    page_icon="📡",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:2.4rem;'>📡 TradePulse</h1>
<h3 style='color:#666; font-weight:400; margin-top:-10px;'>Economic Power Shift Detector — NLP × Geopolitics</h3>
<hr/>
""", unsafe_allow_html=True)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    SUPPORTED_PAIRS = [
        ("IN", "CN"), ("IN", "US"), ("IN", "RU"),
        ("IN", "EU"), ("CN", "US"), ("IN", "BD"),
        ("IN", "JP"), ("IN", "SA"),
    ]
    pair_labels = [f"{COUNTRY_NAMES[a]} ↔ {COUNTRY_NAMES[b]}" for a, b in SUPPORTED_PAIRS]
    selected_idx = st.selectbox("Select Country Pair", range(len(pair_labels)), format_func=lambda i: pair_labels[i])
    country_a, country_b = SUPPORTED_PAIRS[selected_idx]
    name_a, name_b = COUNTRY_NAMES[country_a], COUNTRY_NAMES[country_b]

    max_articles = st.slider("Max articles per feed", 5, 30, 15)
    run_btn = st.button("🔍 Analyse Now", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("""
**Data sources**
- Reuters RSS
- BBC Business RSS
- PIB India RSS
- Al Jazeera Economy
- Economic Times RSS

**NLP Model**
- ProsusAI/FinBERT
- Bilateral framing context
- 3-class: cooperative / neutral / adversarial

**Leverage Formula**
```
L = 0.40 × TradeAsymmetry
  + 0.35 × SentimentDelta
  + 0.25 × StanceDivergence
→ scaled 0–10
```
    """)

# ── Main ──────────────────────────────────────────────────────────────────────
if not run_btn:
    st.info(f"👈 Select a country pair and click **Analyse Now** to run the live NLP pipeline.")
    st.markdown("""
    ### What TradePulse does
    
    Most people hear *"China is rising"* or *"US tariffs on India"* but have no way to see 
    the **underlying pattern**, track it over time, or understand the causal mechanism.
    
    TradePulse combines **three independent signals** into a single interpretable Leverage Score:
    
    | Signal | Source | Method |
    |---|---|---|
    | Trade Dependency Asymmetry | World Bank bilateral trade data | Structural computation |
    | Bilateral News Sentiment | GDELT + RSS feeds | Fine-tuned FinBERT |
    | Policy Stance Divergence | PIB, RBI, USTR press releases | RoBERTa classifier |
    
    The output: a score from **0 (stable)** to **10 (critical leverage imbalance)** with a 
    plain-English explanation of what is driving the change.
    """)
    st.stop()

# ── Live pipeline ─────────────────────────────────────────────────────────────
progress = st.progress(0, text="Starting pipeline...")

# Step 1: Fetch news
progress.progress(10, text=f"📰 Fetching live news for {name_a} ↔ {name_b}...")
with st.spinner("Fetching headlines from Reuters, PIB, BBC, Al Jazeera..."):
    all_articles = fetch_articles(max_per_feed=max_articles)
    pair_articles = filter_for_pair(all_articles, country_a, country_b)

progress.progress(35, text=f"✅ Fetched {len(all_articles)} articles, {len(pair_articles)} relevant")

# Step 2: Trade asymmetry
trade_data = get_trade_asymmetry(country_a, country_b)
progress.progress(50, text="💹 Computing trade asymmetry...")

# Step 3: Sentiment
if pair_articles:
    progress.progress(60, text="🧠 Running FinBERT sentiment analysis...")
    with st.spinner("Running FinBERT on bilateral news articles..."):
        scored = score_articles(pair_articles, name_a, name_b)
    sentiment_data = compute_pair_score(scored)
else:
    # Demo fallback with labelled test headlines
    progress.progress(60, text="🧠 Running FinBERT on sample headlines (no live articles found)...")
    DEMO_HEADLINES = {
        ("IN","CN"): [
            "India imposes anti-dumping duty on Chinese steel imports, escalating trade tensions",
            "India and China agree to ease border tensions, resume bilateral trade",
            "India's trade deficit with China hits record $85 billion this fiscal year",
            "India bans Chinese apps citing national security, tensions escalate",
            "India-China diplomatic talks on trade normalisation show progress",
        ],
        ("IN","US"): [
            "US imposes tariffs on Indian steel and aluminium exports",
            "India and United States sign strategic trade partnership agreement",
            "India-US trade deal discussions advance, market access on agenda",
            "American companies boost investment in India amid supply chain shift",
            "India challenges US tariffs at WTO dispute settlement body",
        ],
        ("IN","RU"): [
            "India continues to buy discounted Russian oil despite Western pressure",
            "India-Russia trade in rupee-ruble settlement faces banking hurdles",
            "Russia remains India's top oil supplier, bilateral trade surges",
            "India diversifies energy imports amid concerns over Russia dependency",
            "India and Russia sign long-term energy cooperation agreement",
        ],
    }
    key = (country_a, country_b) if (country_a, country_b) in DEMO_HEADLINES else ("IN","CN")
    demo_arts = [{"text": h, "title": h, "source": "Demo corpus", "url": ""} for h in DEMO_HEADLINES[key]]
    with st.spinner("Running FinBERT on demo corpus..."):
        scored = score_articles(demo_arts, name_a, name_b)
    sentiment_data = compute_pair_score(scored)
    st.warning("⚠️ No live articles found for this pair today. Using demonstration corpus for NLP showcase.")

# Step 4: Leverage score
progress.progress(90, text="📊 Computing Leverage Signal...")
leverage = compute_leverage_score(country_a, country_b, trade_data, sentiment_data)
interpretation = get_interpretation(leverage["leverage_score"], f"{name_a}↔{name_b}", trade_data, sentiment_data)
progress.progress(100, text="✅ Done!")
progress.empty()

# ── Results layout ────────────────────────────────────────────────────────────
st.markdown(f"## 🌐 {name_a} ↔ {name_b} — Leverage Analysis")

# Score display
col1, col2, col3, col4 = st.columns(4)
score_color = {"Stable": "🟢", "Watchlist": "🟡", "Elevated": "🟠", "Critical": "🔴"}
with col1:
    st.metric("Leverage Score", f"{leverage['leverage_score']} / 10", 
              delta=leverage["label"], delta_color="off")
with col2:
    st.metric("Status", f"{score_color[leverage['label']]} {leverage['label']}")
with col3:
    st.metric("News Sentiment", sentiment_data["label"].upper(),
              delta=f"{sentiment_data['n']} articles")
with col4:
    st.metric("Trade Asymmetry", f"{trade_data['raw_asymmetry']:.1%}")

# Score bar
score_pct = leverage["leverage_score"] / 10
bar_color = {"Stable": "#22c55e", "Watchlist": "#eab308", 
             "Elevated": "#f97316", "Critical": "#ef4444"}[leverage["label"]]
st.markdown(f"""
<div style='margin:12px 0;'>
  <div style='background:#e5e7eb; border-radius:8px; height:28px; width:100%;'>
    <div style='background:{bar_color}; border-radius:8px; height:28px; width:{score_pct*100:.0f}%;
                display:flex; align-items:center; padding-left:10px; color:white; font-weight:bold;'>
      {leverage['leverage_score']} / 10 — {leverage['label']}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Interpretation
st.info(f"**📝 Analysis:** {interpretation}")

# ── Component breakdown ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Component Breakdown")
c = leverage["components"]

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("#### 💹 Trade Asymmetry (40%)")
    st.progress(c["trade_asymmetry"])
    st.markdown(f"**Score:** {c['trade_asymmetry']:.3f}")
    st.markdown(f"**{name_a} → {name_b}:** {trade_data.get('share_a', 'N/A'):.1%}" 
                if trade_data.get('share_a') else f"**Export share:** estimated")
    st.markdown(f"**{name_b} → {name_a}:** {trade_data.get('share_b', 'N/A'):.1%}"
                if trade_data.get('share_b') else "")
    st.caption(trade_data.get("note", ""))

with col2:
    st.markdown("#### 🧠 News Sentiment (35%)")
    st.progress(c["sentiment"])
    st.markdown(f"**Score:** {c['sentiment']:.3f}")
    st.markdown(f"🟢 Cooperative: **{sentiment_data['cooperative']}**")
    st.markdown(f"🟡 Neutral: **{sentiment_data['neutral']}**")
    st.markdown(f"🔴 Adversarial: **{sentiment_data['adversarial']}**")
    st.caption(f"Mean sentiment: {sentiment_data['mean']:+.3f}")

with col3:
    st.markdown("#### 🏛️ Policy Stance (25%)")
    st.progress(c["stance_divergence"])
    st.markdown(f"**Score:** {c['stance_divergence']:.3f}")
    st.caption("Phase 2 feature — RoBERTa classifier on PIB/RBI/USTR press releases. "
               "Currently using neutral default (0.5). Will be live after fine-tuning.")

# ── Article breakdown ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### 📰 Articles Analysed ({len(scored)} total)")

label_emoji = {"cooperative": "🟢", "neutral": "🟡", "adversarial": "🔴"}
label_color = {"cooperative": "#dcfce7", "neutral": "#fef9c3", "adversarial": "#fee2e2"}

for art in scored[:15]:
    s = art["sentiment"]
    emoji = label_emoji[s["label"]]
    bg    = label_color[s["label"]]
    with st.expander(f"{emoji} {art['title'][:90]}...  [{s['label'].upper()} | conf: {s['confidence']:.2f}]"):
        col1, col2 = st.columns([3,1])
        with col1:
            st.markdown(f"**Source:** {art['source']}")
            st.markdown(f"**URL:** {art.get('url','N/A')}")
        with col2:
            st.markdown(f"**Label:** `{s['label']}`")
            st.markdown(f"**Confidence:** `{s['confidence']}`")
            for lbl, sc in s["raw_scores"].items():
                st.markdown(f"- {lbl}: `{sc:.3f}`")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#666; font-size:0.85rem;'>
TradePulse — NLP Project Demo | Phase 1 & 2 | 
Data: World Bank · Reuters · PIB India · BBC · Al Jazeera | 
Model: ProsusAI/FinBERT | Fine-tuning: Phase 2 →
</div>
""", unsafe_allow_html=True)
