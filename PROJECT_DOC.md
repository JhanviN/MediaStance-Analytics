# MediaStance Analytics — Complete Project Documentation

## What Is This Project?

MediaStance Analytics is a real-time bilateral geopolitical sentiment classification system. It reads news headlines and classifies the relationship between two countries as **adversarial**, **cooperative**, or **neutral**.

The system covers 15 bilateral pairs across 7 countries: India (IN), China (CN), USA (US), Russia (RU), Pakistan (PK), Iran (IR), Israel (IL).

**Core use case:** Instead of reading hundreds of headlines daily, analysts get a dashboard showing whether the India-China relationship is trending adversarial this week vs last week, which headlines are driving the signal, and when adversarial sentiment spikes above a threshold.

---

## Architecture Overview

```
RSS Feeds + GDELT + Synthetic Data
            ↓
    Data Pipeline (collect → label → merge)
            ↓
    NLP Models (TF-IDF baseline + DistilBERT)
            ↓
    Predictions Database (SQLite)
            ↓
    FastAPI Analytics Layer
            ↓
    Streamlit Dashboard + Live Pipeline
```

---

## What Has Been Implemented

### 1. Data Collection Pipeline

**Files:** `scripts/collect_corpus.py`, `core/news_fetcher.py`, `nlp/supplemental_feeds.py`

- Fetches from 6 general RSS feeds (Reuters, BBC, Al Jazeera, Economic Times, PIB India)
- 15 Google News search RSS feeds — one per bilateral pair
- 90-day rolling window, configurable per run
- Deduplication via Jaccard word overlap (catches near-duplicate headlines)
- UTF-8 BOM output for Excel compatibility
- `--merge` flag preserves existing rows across multiple runs

**Output:** `data/raw_headlines.csv` with columns: id, headline, country_1, country_2, source, url, published_at, text

### 2. GDELT Real-World Data

**File:** `scripts/fetch_gdelt.py`

- Downloads GDELT 2.0 event files (15-minute intervals, public domain)
- Maps CAMEO event codes to labels: codes 1-8 → cooperative, 9-11 → neutral, 12-20 → adversarial
- Deduplicates by URL+pair, keeps dominant label per article
- Parallel enrichment (20 workers) fetches real article titles from source URLs
- Stratified enrichment: 300 rows per (pair, label) bucket — prevents dominant pairs from monopolizing
- **Result:** 52,381 raw rows fetched, 9,013 enriched with real headlines

**CAMEO taxonomy:** 30-year political science standard developed at Harvard. Assigns structured event codes to news events globally.

### 3. Synthetic Data Generation

**File:** `scripts/generate_synthetic.py`

- Supports LM Studio (local), OpenAI, and Gemini backends
- Pair-specific prompts with domain context (e.g., IN-RU focuses on oil imports, S-400, sanctions)
- Three label classes generated separately with tailored instructions
- Incremental saving after every (pair, label) combo — crash-safe
- Skip logic on restart: detects already-completed combos, resumes from where it left off
- Deduplication by content hash
- **Result:** 11,672 synthetic headlines across all 15 pairs, balanced labels

**Why synthetic data is ethical:** Standard NLP practice (Stanford Alpaca, GPT fine-tuning datasets). Disclosed transparently. Evaluated separately on human-only gold test set.

### 4. Data Merging and Balancing

**File:** `scripts/merge_augmented_data.py`

Priority order:
1. Human labels — always kept, never capped
2. GDELT — capped at 50 per (pair, label) due to label noise
3. Synthetic — fills remaining gaps

**Result:** 13,925 merged rows, near-perfect label balance (33%/34%/33%)

### 5. NLP Models

**Baseline — TF-IDF + Logistic Regression**
- File: `scripts/train_baseline.py`
- 30,000 features, bigrams, sublinear TF scaling
- Balanced class weights
- Fast, interpretable, strong baseline

**Advanced — DistilBERT Fine-tuning**
- File: `scripts/train_transformer.py`
- Pre-trained: `distilbert-base-uncased`
- Entity-aware input: pair prepended to headline (`IN-CN: headline text`)
- 5 epochs, batch size 32, fp16 on GPU
- Early stopping via `load_best_model_at_end=True`
- Optimized for macro F1, not accuracy
- Trained on Google Colab T4 GPU (~18 minutes)

**Entity-aware encoding:** The pair is injected into the model input so the same headline gets different representations for different pairs. "India and China hold talks" means something different for IN-CN vs IN-US queries.

### 6. Evaluation

**Files:** `scripts/evaluate.py`, `scripts/evaluate_human.py`, `scripts/plot_training.py`

| Metric | Baseline | Transformer |
|---|---|---|
| Mixed test accuracy | 87.54% | 87.86% |
| Mixed test macro F1 | 87.50% | 87.84% |
| Human gold accuracy | 79.64% | 78.93% |
| Human gold macro F1 | 78.89% | 78.03% |
| AUC adversarial | 0.979 | 0.982 |
| AUC cooperative | 0.977 | 0.981 |
| AUC neutral | 0.964 | 0.968 |

**Two test sets:**
- Mixed (2,785 rows): 87% synthetic + 13% real — measures task learning
- Human gold (280 rows): 100% human-labeled — honest real-world benchmark

**Training plots generated:** loss curves, confusion matrices, per-class metrics, ROC curves, precision-recall curves

### 7. Error Analysis

**File:** `scripts/error_analysis.py`, `results/error_analysis.md`

338 misclassifications categorized into three types:
1. **Genuine ambiguity (~40%):** Even humans would disagree. Sanctions labeled neutral by human, adversarial by model — both defensible.
2. **Data noise (~30%):** GDELT linked wrong article to a pair. Headline has no bilateral signal.
3. **Soft signal detection (~30%):** Model misses subtle cooperative language ("first steps toward", "strengthening ties").

### 8. Predictions Database

**File:** `nlp/predictions_sqlite.py`

SQLite database with indexed queries:
- `predictions` table: id, created_at, headline, text_used, country_1, country_2, model, label, confidence, p_adversarial, p_cooperative, p_neutral
- Indices on: created_at, label, (country_1, country_2), model
- **Current state:** 18,076 predictions (9,064 baseline + 9,012 transformer)

### 9. FastAPI Serving Layer

**Files:** `api/main.py`, `api/routes_analytics.py`

Endpoints:
- `POST /classify` — single headline classification
- `POST /classify/batch` — up to 50 headlines
- `GET /summary?pair=IN-US` — label distribution for a pair
- `GET /summary/all` — all 15 pairs at once
- `GET /trends?pair=CN-US&rolling=7` — daily time series with rolling average
- `GET /headlines?pair=IL-IR&label=adversarial` — browse predictions
- `GET /alerts` — adversarial spike detection
- `GET /compare?pair1=IN-US&pair2=CN-US` — side-by-side comparison
- `GET /health` — health check

Run: `uvicorn api.main:app --reload --host 127.0.0.1 --port 8000`
Docs: `http://127.0.0.1:8000/docs`

### 10. Streamlit Dashboard

**File:** `demo/demo_app.py`

6 pages:
- **Overview:** All 15 pairs with color-coded adversarial/cooperative/neutral percentages + active alerts
- **Trends:** Daily sentiment timeline with rolling average, distribution donut chart
- **Alerts:** Configurable adversarial spike detection with severity levels
- **Headlines:** Browse predictions by pair, label, date, confidence
- **Compare Pairs:** Side-by-side bar chart comparison
- **Live Predict:** Real-time classification of any headline

Run: `streamlit run demo/demo_app.py`

### 11. Live Pipeline

**File:** `scripts/live_pipeline.py`

Runs continuously:
1. Fetches fresh headlines from all RSS feeds + Google News (last 24h)
2. Runs through trained model
3. Saves predictions to database
4. Repeats every N minutes

Run: `python scripts/live_pipeline.py --model baseline --interval 60`
Single run: `python scripts/live_pipeline.py --once`

---

## Data Flow Summary

```
collect_corpus.py → raw_headlines.csv
fetch_gdelt.py    → gdelt_raw.csv (enriched)
generate_synthetic.py → synthetic_raw.csv
        ↓
merge_augmented_data.py → labeled_dataset_augmented.csv
        ↓
split_data.py → train.csv + test.csv
        ↓
train_baseline.py → models/baseline_tfidf_lr.joblib
train_transformer.py → models/transformer_bilateral/
        ↓
predict_batch.py → predictions.db (18,076 rows)
        ↓
demo_app.py (dashboard) + api/main.py (REST API)
        ↓
live_pipeline.py (continuous updates)
```

---

## Key Technical Decisions

**Why entity-aware input?**
Without it, the model sees the same headline regardless of which pair is being queried. "India and China hold talks" gets identical encoding for IN-CN and IN-US. Prepending the pair (`IN-CN: headline`) teaches the model which relationship to evaluate.

**Why macro F1 over accuracy?**
Accuracy rewards majority-class prediction. Macro F1 penalizes poor performance on any single class equally. For a 3-class balanced problem, macro F1 is the honest metric.

**Why two test sets?**
The mixed test set (87% synthetic) measures task learning. The human gold set (100% human-labeled) measures real-world generalization. Reporting only the mixed set would be misleading.

**Why GDELT at cap 50?**
GDELT's CAMEO codes are assigned to events, not articles. One article can generate multiple events with conflicting codes. At ~40% label noise rate, capping at 50 per bucket limits noise contribution while preserving real-world text variety.

**Why synthetic data?**
GDELT underrepresents India-centric pairs (IN-PK, IN-IR, IN-CN) because English news volume for those pairs is lower than active conflict pairs (IR-US, IL-IR). Synthetic generation fills this gap with pair-specific, contextually accurate headlines.

---

## What Is Left / Future Work

### High Priority
- **Confidence calibration:** ✅ DONE — Temperature scaling implemented (T=1.5276, ECE improved 46%). Saved to `models/transformer_bilateral/temperature.json`, auto-applied in inference.
- **Inter-annotator agreement:** Get a second person to label 100 headlines independently. Measure Cohen's kappa. If kappa > 0.6, the task definition is validated.
- **Daily scheduler:** Automate `live_pipeline.py` via cron job or Windows Task Scheduler for truly autonomous operation.

### Medium Priority
- **Attention visualization:** ✅ DONE — Live in dashboard (🧠 Attention page). Token heatmap, per-class probabilities, top words summary per class. Reveals model focuses on "tariffs/sanctions/retaliation" for adversarial and "agreement/deal/partnership" for cooperative.
- **Causality graph:** GDELT stores event sequences. Tracing "what events caused this adversarial spike" is the next analytical layer. Requires storing CAMEO codes alongside labels.
- **Macro context panel:** Integrate the geopolitical risk dataset (political stability index, GDP growth, military expenditure) as contextual features alongside NLP predictions.

### Lower Priority
- **API authentication:** Rate limiting and API keys for production deployment
- **Model versioning:** Track model versions in the database so predictions are tied to the model that made them
- **Confidence threshold filtering:** Only show predictions above a minimum confidence in the dashboard

---

## Project Stats

| Item | Count |
|---|---|
| Country pairs | 15 |
| Countries | 7 |
| Training rows | 11,140 |
| Test rows | 2,785 |
| Human gold test rows | 280 |
| Predictions in DB | 18,076 |
| Transformer macro F1 (mixed) | 87.84% |
| Transformer macro F1 (human gold) | 78.03% |
| AUC range | 0.964 – 0.982 |
| API endpoints | 9 |
| Dashboard pages | 6 |

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Collect fresh data
python scripts/collect_corpus.py --feeds 50 --merge

# Start live pipeline (continuous)
python scripts/live_pipeline.py --model baseline --interval 60

# Start dashboard
streamlit run demo/demo_app.py

# Start API
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

# Classify a headline
python scripts/predict.py -t "India and China sign trade deal" --pair IN-CN
```
