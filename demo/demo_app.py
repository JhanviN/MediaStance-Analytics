"""
TradePulse — Streamlit Dashboard
Run: streamlit run demo/demo_app.py
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.predictions_sqlite import connect, init_predictions_db
from nlp.analytics_queries import (
    label_distribution, summary_all_pairs, trends_by_day,
    rolling_trends, list_headlines, detect_spikes
)
from nlp.pair_utils import CANONICAL_PAIR_KEYS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TradePulse",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.adversarial { color: #F44336; font-weight: bold; }
.cooperative { color: #4CAF50; font-weight: bold; }
.neutral     { color: #2196F3; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

LABEL_COLORS = {
    "adversarial": "#F44336",
    "cooperative": "#4CAF50",
    "neutral":     "#2196F3",
}

# ── DB connection ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_conn():
    db = ROOT / "data" / "predictions.db"
    conn = connect(db)
    init_predictions_db(conn)
    return conn

conn = get_conn()

# Check if DB has data
total = conn.execute("SELECT COUNT(*) as c FROM predictions").fetchone()["c"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🌐 TradePulse")
st.sidebar.caption("Bilateral Geopolitical Sentiment")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📈 Trends", "🔴 Alerts", "📰 Headlines", "⚖️ Compare Pairs", "🔮 Live Predict"]
)

model = st.sidebar.selectbox("Model", ["baseline", "transformer"], index=0)
all_pairs = sorted(CANONICAL_PAIR_KEYS)

st.sidebar.markdown("---")
st.sidebar.caption(f"Total predictions in DB: **{total:,}**")
st.sidebar.caption(f"Model: **{model}**")

# ── Helper ────────────────────────────────────────────────────────────────────
def pair_selector(key="pair", default="CN-US"):
    idx = all_pairs.index(default) if default in all_pairs else 0
    return st.selectbox("Select Pair", all_pairs, index=idx, key=key)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🌐 TradePulse — Geopolitical Sentiment Dashboard")
    st.caption("Real-time bilateral relationship classification from news headlines")

    if total == 0:
        st.warning("No predictions in database yet. Run: `python scripts/predict_batch.py --input data/gdelt_raw.csv --model baseline`")
        st.stop()

    st.markdown("### All Pairs Summary")
    summary = summary_all_pairs(conn, model=model)

    cols = st.columns(3)
    pairs_data = summary.get("pairs", {})

    for i, (pair_key, dist) in enumerate(sorted(pairs_data.items())):
        col = cols[i % 3]
        with col:
            total_pair = dist.get("total", 0)
            if total_pair == 0:
                continue
            pct = dist.get("percent", {})
            adv = pct.get("adversarial", 0)
            coop = pct.get("cooperative", 0)
            neut = pct.get("neutral", 0)
            dominant = max(pct, key=pct.get) if pct else "neutral"
            color = LABEL_COLORS[dominant]

            st.markdown(f"""
            <div style='border-left: 4px solid {color}; padding: 10px; margin: 5px 0; background: #0e1117; border-radius: 4px;'>
                <b style='font-size:16px'>{pair_key}</b><br>
                <span style='color:#F44336'>▼ {adv:.1f}%</span> &nbsp;
                <span style='color:#4CAF50'>▲ {coop:.1f}%</span> &nbsp;
                <span style='color:#2196F3'>● {neut:.1f}%</span><br>
                <small style='color:#888'>{total_pair} predictions</small>
            </div>
            """, unsafe_allow_html=True)

    # Alerts
    st.markdown("---")
    st.markdown("### 🚨 Active Alerts")
    alerts = detect_spikes(conn, model=model, threshold_pp=15.0, days=7)
    if not alerts:
        st.success("No adversarial spikes detected in the last 7 days.")
    else:
        for alert in alerts:
            sev_color = "#F44336" if alert["severity"] == "high" else "#FF9800"
            st.markdown(f"""
            <div style='border-left: 4px solid {sev_color}; padding: 10px; margin: 5px 0; background: #1a0a0a; border-radius: 4px;'>
                <b>{alert['pair']}</b> — {alert['severity'].upper()} severity<br>
                {alert['message']}<br>
                <small>Window: {alert['window_days']} days | This: {alert['this_window_n']} | Prior: {alert['prev_window_n']}</small>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Trends
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Trends":
    st.title("📈 Sentiment Trends")

    try:
        import plotly.graph_objects as go
        import plotly.express as px
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False
        st.warning("Install plotly for interactive charts: `pip install plotly`")

    pair = pair_selector("trends_pair", "CN-US")
    rolling_window = st.slider("Rolling average window (days)", 1, 14, 7)

    series = trends_by_day(conn, pair, model=model)

    if not series:
        st.info(f"No trend data for {pair} with model={model}. Run batch prediction first.")
    else:
        if HAS_PLOTLY:
            fig = go.Figure()
            dates = [s["date"] for s in series]
            for label, color in LABEL_COLORS.items():
                vals = [s[label] * 100 for s in series]
                fig.add_trace(go.Scatter(
                    x=dates, y=vals, name=label.capitalize(),
                    line=dict(color=color, width=2),
                    mode="lines+markers", marker=dict(size=4),
                ))

            if rolling_window > 1:
                rolling = rolling_trends(series, window=rolling_window, field="adversarial")
                if rolling:
                    fig.add_trace(go.Scatter(
                        x=[r["date"] for r in rolling],
                        y=[r["adversarial_rolling_avg"] * 100 for r in rolling],
                        name=f"Adversarial {rolling_window}d avg",
                        line=dict(color="#FF5722", width=3, dash="dash"),
                    ))

            fig.update_layout(
                title=f"{pair} — Daily Sentiment Distribution",
                xaxis_title="Date", yaxis_title="Share (%)",
                yaxis=dict(range=[0, 100]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback table
            st.dataframe(series)

        # Distribution donut
        dist = label_distribution(conn, pair, model=model)
        if dist["total"] > 0 and HAS_PLOTLY:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**{pair} overall distribution**")
                fig2 = go.Figure(go.Pie(
                    labels=[k.capitalize() for k in dist["counts"].keys()],
                    values=list(dist["counts"].values()),
                    marker_colors=[LABEL_COLORS[k] for k in dist["counts"].keys()],
                    hole=0.4,
                ))
                fig2.update_layout(height=300, showlegend=True, margin=dict(t=20, b=20))
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                st.metric("Total predictions", dist["total"])
                for lab, pct in dist["percent"].items():
                    st.metric(lab.capitalize(), f"{pct:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Alerts
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔴 Alerts":
    st.title("🚨 Adversarial Spike Alerts")

    col1, col2 = st.columns(2)
    threshold = col1.slider("Alert threshold (% point rise)", 5, 40, 15)
    days = col2.slider("Window (days)", 3, 30, 7)

    alerts = detect_spikes(conn, model=model, threshold_pp=threshold, days=days)

    if not alerts:
        st.success(f"No adversarial spikes above {threshold}pp in the last {days} days.")
    else:
        st.error(f"{len(alerts)} alert(s) detected")
        for alert in alerts:
            with st.expander(f"🔴 {alert['pair']} — {alert['severity'].upper()}"):
                st.write(alert["message"])
                col1, col2, col3 = st.columns(3)
                col1.metric("Delta (pp)", f"+{alert['delta_adversarial_pp']:.1f}")
                col2.metric("This window", alert["this_window_n"])
                col3.metric("Prior window", alert["prev_window_n"])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Headlines
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📰 Headlines":
    st.title("📰 Headlines Browser")

    col1, col2, col3 = st.columns(3)
    pair = col1.selectbox("Pair", all_pairs, key="hl_pair")
    label_filter = col2.selectbox("Label", ["all", "adversarial", "cooperative", "neutral"])
    sort_by = col3.selectbox("Sort by", ["date", "confidence"])

    label_arg = None if label_filter == "all" else label_filter
    headlines = list_headlines(conn, pair, model=model, label=label_arg, sort=sort_by, limit=100)

    if not headlines:
        st.info(f"No headlines found for {pair}.")
    else:
        st.caption(f"Showing {len(headlines)} headlines")
        for h in headlines:
            color = LABEL_COLORS.get(h["label"], "#888")
            conf_bar = "█" * int(h["confidence"] * 10) + "░" * (10 - int(h["confidence"] * 10))
            st.markdown(f"""
            <div style='border-left: 3px solid {color}; padding: 8px 12px; margin: 4px 0; background: #0e1117; border-radius: 4px;'>
                <span style='color:{color}; font-size:11px; font-weight:bold'>{h['label'].upper()}</span>
                <span style='color:#888; font-size:11px; margin-left:8px'>{conf_bar} {h['confidence']:.2f}</span>
                <span style='color:#555; font-size:11px; float:right'>{h['created_at'][:10]}</span><br>
                <span style='font-size:14px'>{h['headline'][:150]}</span>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Compare Pairs
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Compare Pairs":
    st.title("⚖️ Compare Two Pairs")

    try:
        import plotly.graph_objects as go
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False

    col1, col2 = st.columns(2)
    pair1 = col1.selectbox("Pair 1", all_pairs, index=all_pairs.index("CN-US") if "CN-US" in all_pairs else 0)
    pair2 = col2.selectbox("Pair 2", all_pairs, index=all_pairs.index("IN-US") if "IN-US" in all_pairs else 1)

    d1 = label_distribution(conn, pair1, model=model)
    d2 = label_distribution(conn, pair2, model=model)

    if HAS_PLOTLY and d1["total"] > 0 and d2["total"] > 0:
        labels = ["Adversarial", "Cooperative", "Neutral"]
        fig = go.Figure(data=[
            go.Bar(name=pair1, x=labels,
                   y=[d1["percent"]["adversarial"], d1["percent"]["cooperative"], d1["percent"]["neutral"]],
                   marker_color=["#F44336", "#4CAF50", "#2196F3"]),
            go.Bar(name=pair2, x=labels,
                   y=[d2["percent"]["adversarial"], d2["percent"]["cooperative"], d2["percent"]["neutral"]],
                   marker_color=["#FF8A80", "#B9F6CA", "#82B1FF"]),
        ])
        fig.update_layout(
            barmode="group", title=f"{pair1} vs {pair2} — Label Distribution",
            yaxis_title="Percentage (%)", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    for col, pair_key, dist in [(col1, pair1, d1), (col2, pair2, d2)]:
        with col:
            st.markdown(f"**{pair_key}** — {dist['total']} predictions")
            for lab in ["adversarial", "cooperative", "neutral"]:
                pct = dist["percent"].get(lab, 0)
                color = LABEL_COLORS[lab]
                st.markdown(f"<span style='color:{color}'>{lab.capitalize()}: **{pct:.1f}%**</span>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Live Predict
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Live Predict":
    st.title("🔮 Live Prediction")
    st.caption("Classify any headline in real-time")

    from nlp.inference import predict_baseline, predict_transformer

    headline = st.text_area("Enter headline", placeholder="India and China sign new trade agreement...", height=80)
    pair_input = st.selectbox("Bilateral pair", all_pairs, index=all_pairs.index("CN-IN") if "CN-IN" in all_pairs else 0)
    run_model = st.selectbox("Model", ["both", "baseline", "transformer"])

    if st.button("🔍 Classify", type="primary") and headline.strip():
        c1, c2 = pair_input.split("-")
        input_text = f"{c1}-{c2}: {headline.strip()}"

        with st.spinner("Classifying..."):
            results = {}
            if run_model in ("baseline", "both"):
                lab, conf, probs = predict_baseline(input_text)
                results["Baseline"] = (lab, conf, probs)
            if run_model in ("transformer", "both"):
                lab, conf, probs = predict_transformer(input_text)
                results["Transformer"] = (lab, conf, probs)

        for model_name, (lab, conf, probs) in results.items():
            color = LABEL_COLORS.get(lab, "#888")
            st.markdown(f"""
            <div style='border: 2px solid {color}; border-radius: 10px; padding: 16px; margin: 8px 0;'>
                <h3 style='color:{color}; margin:0'>{lab.upper()}</h3>
                <p style='margin:4px 0'><b>{model_name}</b> — Confidence: <b>{conf*100:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

            try:
                import plotly.graph_objects as go
                fig = go.Figure(go.Bar(
                    x=[k.capitalize() for k in probs.keys()],
                    y=[v * 100 for v in probs.values()],
                    marker_color=[LABEL_COLORS[k] for k in probs.keys()],
                ))
                fig.update_layout(
                    title=f"{model_name} — Probability Distribution",
                    yaxis_title="%", yaxis_range=[0, 100], height=250,
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.json(probs)
