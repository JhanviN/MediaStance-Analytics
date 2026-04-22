# MediaStance Analytics — Complete Project Documentation

## What Is This Project?

MediaStance Analytics is a real-time bilateral geopolitical sentiment classification system. It reads news headlines and classifies the relationship between two countries as **adversarial**, **cooperative**, or **neutral**.

The system covers 15 bilateral pairs across 7 countries: India (IN), China (CN), USA (US), Russia (RU), Pakistan (PK), Iran (IR), Israel (IL).

**Core use case:** Instead of reading hundreds of headlines daily, analysts get a dashboard showing whether the India-China relationship is trending adversarial this week vs last week, which headlines are driving the signal, and when adversarial sentiment spikes above a threshold.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  DATA SOURCES                        │
│  RSS Feeds (live)  │  GDELT (historical)  │ Synthetic│
└────────────────────┴──────────────────────┴──────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│              DATA PIPELINE                           │
│  collect → label → merge → split → train            │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│              NLP MODELS                              │
│  TF-IDF + LR (baseline)  │  DistilBERT (advanced)   │
└──────────────────────────┴──────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│           PREDICTIONS DATABASE (SQLite)              │
│  18,076 predictions │ 15 pairs │ 2 models            │
└─────────────────────────────────────────────────────┘
                          ↓
          ┌───────────────┴───────────────┐
          ↓                               ↓
┌─────────────────┐             ┌─────────────────────┐
│  FastAPI (REST) │             │ Streamlit Dashboard  │
│  9 endpoints    │             │ 7 pages              │
└─────────────────┘             └─────────────────────┘
          ↑
┌─────────────────────────────────────────────────────┐
│           LIVE PIPELINE (APScheduler)                │
│  RSS → predict → DB  (every 60 min, auto-scheduled) │
└─────────────────────────────────────────────────────┘
```

---

## Data Sources — How Each Is Used

| Source | Purpose | Volume | Notes |
|---|---|---|---|
| RSS feeds | Live real-time headlines | ~50 articles/run | Used by live pipeline every 60 min |
| GDELT | Historical labeled data for training | 9,013 enriched rows | Used once for DB seeding + training |
| Synthetic (LM Studio) | Training data augmentation | 11,672 rows | Fills gaps in underrepresented pairs |
| Human labels | Ground truth + gold test set | 280 rows | Never used for training, only evaluation |

**Key distinction:** RSS is for live predictions. GDELT was used to build the historical baseline. They serve different purposes.

---

## What Has Been Implemented

### 1. Data Collection Pipeline

**Files:** `scripts/collect_corpus.py`, `core/news_fetcher.py`, `nlp/supplemental_feeds.py`

- 6 general RSS feeds (Reuters, BBC, Al Jazeera, Economic Times, PIB India)
- 15 Google News search RSS feeds — one per bilateral pair
- 90-day rolling window, configurable per run
- Deduplication via Jaccard word overlap
- `--merge` flag preserves existing rows across multiple runs

### 2. GDELT Historical Data

**File:** `scripts/fetch_gdelt.py`

- Downloads GDELT 2.0 event files (15-minute intervals, public domain)
- Maps CAMEO event codes → labels: 1-8 cooperative, 9-11 neutral, 12-20 adversarial
- Parallel enrichment (20 workers) fetches real article titles from source URLs
- Stratified enrichment: equal rows per (pair, label) bucket
- **Result:** 52,381 raw rows, 9,013 enriched with real headlines
- **Used for:** seeding predictions.db with historical data + training data

**CAMEO taxonomy:** 30-year political science standard. Assigns structured event codes to news events globally.

### 3. Synthetic Data Generation

**File:** `scripts/generate_synthetic.py`

- Supports LM Studio (local), OpenAI, Gemini backends
- Pair-specific prompts with domain context
- Incremental saving — crash-safe, resumes on restart
- **Result:** 11,672 synthetic headlines, all 15 pairs, balanced labels
- **Used for:** filling training data gaps in India-centric pairs underrepresented in GDELT

### 4. Data Merging and Balancing

**File:** `scripts/merge_augmented_data.py`

Priority: Human labels (always kept) → GDELT (capped at 50/bucket, noisy) → Synthetic (fills gaps)

**Result:** 13,925 merged rows, near-perfect label balance (33%/34%/33%)

### 5. NLP Models

**Baseline — TF-IDF + Logistic Regression** (`scripts/train_baseline.py`)
- 30,000 features, bigrams, sublinear TF, balanced class weights

**Advanced — DistilBERT Fine-tuning** (`scripts/train_transformer.py`)
- Pre-trained: `distilbert-base-uncased`
- Entity-aware input: `IN-CN: headline text` — pair prepended so model knows which relationship to evaluate
- 5 epochs, batch size 32, fp16, Colab T4 GPU (~18 min)
- Optimized for macro F1

### 6. Confidence Calibration ✅

**File:** `scripts/calibrate_model.py`

Temperature scaling (T=1.5276) applied after training:
- ECE before: 0.0647 → ECE after: 0.0346 (46% improvement)
- Mean confidence: 0.9432 → 0.8990 (more honest)
- Saved to `models/transformer_bilateral/temperature.json`, auto-applied in inference
- Makes `/alerts` confidence thresholds meaningful

### 7. Evaluation

**Files:** `scripts/evaluate.py`, `scripts/evaluate_human.py`, `scripts/plot_training.py`

| Metric | Baseline | Transformer |
|---|---|---|
| Mixed test macro F1 | 87.50% | 87.84% |
| Human gold macro F1 | 78.89% | 78.03% |
| AUC adversarial | 0.979 | 0.982 |
| AUC cooperative | 0.977 | 0.981 |
| AUC neutral | 0.964 | 0.968 |

**Two test sets:**
- Mixed (2,785 rows): 87% synthetic — measures task learning
- Human gold (280 rows): 100% human-labeled — honest real-world benchmark

**Plots:** loss curves, confusion matrices, per-class metrics, ROC curves, PR curves

### 8. Error Analysis

338 misclassifications in 3 categories:
1. **Genuine ambiguity (~40%)** — even humans disagree
2. **Data noise (~30%)** — GDELT linked wrong article to pair
3. **Soft signal detection (~30%)** — model misses indirect adversarial language ("deal still far off", "Strait remains closed")

### 9. Predictions Database

SQLite with 18,076 predictions (9,064 baseline + 9,012 transformer). All analytics queries filter by `model=` — no double-counting.

### 10. FastAPI Serving Layer

**Files:** `api/main.py`, `api/routes_analytics.py`

- `POST /classify` — single headline
- `POST /classify/batch` — up to 50 headlines
- `GET /summary?pair=IN-US` — label distribution
- `GET /summary/all` — all 15 pairs
- `GET /trends?pair=CN-US&rolling=7` — time series
- `GET /headlines?pair=IL-IR&label=adversarial` — browse
- `GET /alerts` — spike detection
- `GET /compare?pair1=IN-US&pair2=CN-US` — comparison
- `GET /health` — health check

**APScheduler built-in:** API starts a background scheduler that runs the live pipeline every 60 minutes automatically. Works on any deployment — local, AWS, DigitalOcean, Railway.

### 11. Streamlit Dashboard

**File:** `demo/demo_app.py` — 7 pages:

| Page | What it shows |
|---|---|
| 🏠 Overview | All 15 pairs, color-coded sentiment, active alerts |
| 📈 Trends | Daily timeline, rolling average, distribution donut |
| 🔴 Alerts | Adversarial spike detection, configurable threshold |
| 📰 Headlines | Browse by pair/label/date/confidence |
| ⚖️ Compare Pairs | Side-by-side bar chart |
| 🔮 Live Predict | Real-time classification of any headline |
| 🧠 Attention | Token attention heatmap, top words per class |

### 12. Attention Visualization ✅

**File:** `scripts/attention_viz.py`

- Extracts CLS token attention from all 6 DistilBERT layers, averaged across 12 heads
- Live heatmap in dashboard — type any headline, see which tokens the model focuses on
- Top words summary per class (adversarial/cooperative/neutral)
- Demonstrates model learned correct semantic signals

### 13. Live Pipeline

**File:** `scripts/live_pipeline.py`

- Fetches fresh headlines from RSS + Google News (last 24h only)
- Runs through trained model, saves to predictions.db
- Runs every 60 minutes via APScheduler (embedded in FastAPI) or Windows Task Scheduler
- `--once` flag for single run (useful for demos)

**Data source for live pipeline: RSS only** (not GDELT). RSS gives real-time headlines within minutes. GDELT has 15-minute delay + requires URL enrichment — not suitable for live use.

---

## Data Flow

```
RSS + Google News → live_pipeline.py → predictions.db → dashboard/API
GDELT (historical) → fetch_gdelt.py → gdelt_raw.csv → predict_batch.py → predictions.db
Synthetic → generate_synthetic.py → synthetic_raw.csv ↘
Human labels → labeled_dataset.csv                    → merge → train → models/
```

---

## Key Technical Decisions

**Entity-aware input:** Prepending `IN-CN:` to headlines teaches the model which relationship to evaluate. Same headline = different encoding for different pairs.

**Macro F1 over accuracy:** Accuracy rewards majority-class prediction. Macro F1 penalizes poor performance on any class equally.

**Two test sets:** Mixed set measures task learning. Human gold set measures real-world generalization. Both reported transparently.

**GDELT capped at 50/bucket:** ~40% label noise rate. Small contribution preserves text variety without corrupting training signal.

**Synthetic data:** GDELT underrepresents India-centric pairs. Synthetic fills gaps with pair-specific context. Standard NLP practice, disclosed transparently.

**Temperature scaling:** Softmax overconfidence corrected. ECE improved 46%. Confidence scores now meaningful for alert thresholds.

---

## What Is Left

### Remaining
- **Inter-annotator agreement:** Label 100 headlines with a second person, measure Cohen's kappa
- **Causality graph:** Store CAMEO codes, trace event sequences that caused adversarial spikes
- **Macro context panel:** Integrate geopolitical risk dataset (political stability, GDP, military expenditure)
- **API authentication:** Rate limiting and API keys for public deployment

### Already Done
- ✅ Data pipeline (RSS + GDELT + synthetic)
- ✅ Two trained models (baseline + DistilBERT)
- ✅ Entity-aware input encoding
- ✅ Confidence calibration (temperature scaling)
- ✅ Full evaluation (mixed + human gold, ROC, PR curves)
- ✅ Error analysis
- ✅ FastAPI with 9 endpoints
- ✅ SQLite predictions database (18,076 rows)
- ✅ Streamlit dashboard (7 pages)
- ✅ Attention visualization (live in dashboard)
- ✅ Live pipeline (RSS → predict → DB)
- ✅ APScheduler (auto-runs in API process, deployment-ready)
- ✅ Windows Task Scheduler setup script

---

## Project Stats

| Item | Value |
|---|---|
| Countries | 7 |
| Bilateral pairs | 15 |
| Training rows | 11,140 |
| Test rows | 2,785 |
| Human gold test rows | 280 |
| Predictions in DB | 18,076 |
| Transformer macro F1 (mixed) | 87.84% |
| Transformer macro F1 (human gold) | 78.03% |
| AUC range | 0.964 – 0.982 |
| Calibration ECE improvement | 46% |
| API endpoints | 9 |
| Dashboard pages | 7 |

---

## How to Run

```bash
pip install -r requirements.txt

# Dashboard
streamlit run demo/demo_app.py

# API (with auto-scheduler)
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

# Live pipeline (manual)
python scripts/live_pipeline.py --model baseline --interval 60

# Single prediction
python scripts/predict.py -t "India and China sign trade deal" --pair IN-CN

# Setup Windows auto-scheduler
python scripts/setup_scheduler.py --interval 60
```
