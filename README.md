# 📡 TradePulse — Economic Power Shift Detector

> NLP × Geopolitics × Public Journalism | Phase 1 & 2 Demo

## What it does

TradePulse detects shifts in bilateral economic leverage between nations by combining:
1. **Trade Dependency Asymmetry** — World Bank bilateral trade data  
2. **Bilateral News Sentiment** — FinBERT fine-tuned on economic news  
3. **Policy Stance Divergence** — RoBERTa on govt press releases (Phase 2)

Output: A **Leverage Score (0–10)** per country pair with plain-English explanation.

## Quick Start

### Option A: Google Colab (recommended for demo)
1. Open `TradePulse_Demo.ipynb` in Google Colab
2. Set Runtime → T4 GPU
3. Run all cells in order
4. See live FinBERT output in Cell 3 + 4

### Option B: Local Streamlit app
```bash
pip install -r requirements.txt
streamlit run demo/demo_app.py
```

## Project Structure
```
tradepulse/
├── core/
│   ├── news_fetcher.py      # RSS feed fetcher + country pair filter
│   ├── sentiment.py         # FinBERT bilateral sentiment classifier
│   └── leverage_engine.py   # Leverage Signal formula + scoring
├── demo/
│   └── demo_app.py          # Streamlit live demo app
├── TradePulse_Demo.ipynb    # Colab notebook (self-contained demo)
└── requirements.txt
```

## Supported Country Pairs
India↔China | India↔USA | India↔Russia | India↔EU  
China↔USA | India↔Bangladesh | India↔Japan | India↔Saudi Arabia

## Tech Stack
| Layer | Tool |
|---|---|
| Sentiment NLP | ProsusAI/FinBERT (HuggingFace) |
| NER / routing | spaCy en_core_web_lg |
| News corpus | GDELT 2.0 + RSS (Reuters, PIB, BBC) |
| Data pipeline | Python + feedparser + requests |
| Trade data | World Bank API (static snapshot for demo) |
| Backend | FastAPI (Phase 3) |
| Frontend | React + D3.js (Phase 4) |

## Leverage Formula
```
Leverage(A,B) = 0.40 × TradeAsymmetry(A,B)
              + 0.35 × SentimentDelta(A,B)    ← FinBERT NLP
              + 0.25 × StanceDivergence(A,B)  ← RoBERTa NLP
→ Scaled to 0–10
```

## Roadmap
- [x] Phase 1: Data pipeline + RSS fetcher + corpus design  
- [x] Phase 2 (demo): FinBERT zero-shot bilateral sentiment  
- [ ] Phase 2 (complete): Fine-tune on 3,000 labelled articles  
- [ ] Phase 3: Leverage Signal engine + LLM explanations  
- [ ] Phase 4: React frontend + D3.js choropleth map  
- [ ] Phase 5: User study + ablation + major report  
