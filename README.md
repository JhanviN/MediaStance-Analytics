# MediaStance Analytics

**Real-time bilateral geopolitical sentiment classification from news headlines.**

Given a news headline and a country pair, the system classifies the bilateral relationship as `adversarial`, `cooperative`, or `neutral` — covering 15 pairs across 7 countries.

## Live Demos

| Service | URL |
|---|---|
<!-- | Streamlit Dashboard | [jhanvin-mediastance-analytics.hf.space](https://jhanvin-mediastance-analytics.hf.space) | -->
| Next.js Frontend | [mediastance.netlify.app](https://tourmaline-klepon-02ce4e.netlify.app/) |
| FastAPI Backend | [mediastance-analytics-1.onrender.com/docs](https://mediastance-analytics-1.onrender.com/docs) |

---

## Results

| Model | Mixed Test Macro F1 | Human Gold Macro F1 | AUC (avg) |
|---|---|---|---|
| TF-IDF + Logistic Regression | 87.50% | 78.89% | 0.973 |
| DistilBERT (fine-tuned) | **87.84%** | **78.03%** | **0.977** |

> The human gold test set (280 rows, 100% human-annotated) is the honest benchmark. The mixed test set (2,785 rows, 87% synthetic) measures task learning.

38,000+ predictions in DB · 15 bilateral pairs · 7 countries · Temperature-calibrated confidence (ECE −46%)

---

## Countries & Pairs

**Countries:** India (IN), China (CN), USA (US), Russia (RU), Pakistan (PK), Iran (IR), Israel (IL)

**Pairs (15):** CN-IN · CN-IR · CN-PK · CN-RU · CN-US · IL-IN · IL-IR · IL-US · IN-IR · IN-PK · IN-RU · IN-US · IR-RU · IR-US · RU-US

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  DATA SOURCES                                        │
│  RSS Feeds (live)  │  GDELT (historical)  │ Synthetic│
└────────────────────┴──────────────────────┴──────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  DATA PIPELINE                                       │
│  collect → label → merge → split → train            │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  NLP MODELS                                          │
│  TF-IDF + LR (baseline)  │  DistilBERT (fine-tuned) │
└──────────────────────────┴──────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  PREDICTIONS DATABASE (SQLite, 38k+ rows)            │
│  Persisted in HF Dataset repo — synced every 30 min │
└─────────────────────────────────────────────────────┘
              ↓
┌──────────────────┐        ┌──────────────────────────┐
│  FastAPI (REST)  │        │  Next.js + Streamlit      │
│  9 endpoints     │ ←────→ │  8 dashboard pages        │
└──────────────────┘        └──────────────────────────┘
              ↑
┌─────────────────────────────────────────────────────┐
│  LIVE PIPELINE (GitHub Actions, daily 06:00 UTC)     │
│  RSS → predict (both models) → DB → HF Dataset sync │
└─────────────────────────────────────────────────────┘
```

---

## Dashboard Pages

| Page | Description |
|---|---|
| Overview | All 15 pairs, color-coded sentiment bars, active alerts |
| Trends | Daily time series with 7-day rolling average, date filters |
| Alerts | Adversarial spike detection, configurable threshold and window |
| Headlines | Browse by pair, label, date, confidence |
| Compare Pairs | Side-by-side distribution comparison |
| Live Predict | Real-time classification of any headline |
| Attention | Token attention heatmap (model interpretability) |
| Causality | State transition graph, recurring patterns |

---

## API Endpoints

```
POST /classify              — single headline classification
POST /classify/batch        — up to 50 headlines per request
GET  /summary/all           — distribution across all 15 pairs
GET  /trends?pair=CN-US     — daily time series with rolling average
GET  /headlines?pair=IL-IR  — browse predictions by pair and label
GET  /alerts                — adversarial spike detection
GET  /compare               — side-by-side pair comparison
GET  /causality             — state transition graph data
GET  /health                — health check
```

Interactive docs: `https://mediastance-analytics-1.onrender.com/docs`

---

## Key Technical Decisions

**Entity-aware input encoding**
Pair prefix prepended to every headline: `IN-CN: India and China hold border talks`. This teaches the model which bilateral relationship to evaluate — without it, the same headline gets identical encoding regardless of which pair is queried.

**Three-source data strategy with priority ordering**
Human annotation (highest priority, always kept) → GDELT weak supervision (capped at 50 rows per pair/label bucket due to ~40% noise) → Synthetic augmentation (fills gaps, lowest priority). Implemented in `scripts/merge_augmented_data.py`.

**Two test sets**
Mixed test set (2,785 rows, 87% synthetic) measures task learning. Human gold test set (280 rows, 100% human-labeled) is the honest real-world benchmark. Both reported transparently.

**Temperature scaling**
Post-hoc confidence calibration (T=1.5276). ECE reduced from 0.0647 → 0.0346 (46% improvement). Makes confidence thresholds in `/alerts` meaningful.

**Macro F1 as primary metric**
Penalizes poor performance on any single class equally — the right metric for a balanced 3-class problem.

---

## Deployment

| Component | Platform | Notes |
|---|---|---|
| Streamlit dashboard | HF Spaces (free) | Docker, port 7860 |
| FastAPI backend | Render (free) | Docker, CPU-only PyTorch |
| Next.js frontend | Netlify (free) | Static export |
| Predictions DB | HF Dataset repo | `JhanviN/mediastance-db` |
| Transformer model | HF Model repo | `JhanviN/mediastance-deploy` |
| Live pipeline | GitHub Actions | Daily at 06:00 UTC |

**Persistent storage pattern:** Both HF Spaces and Render use ephemeral containers. `core/db_sync.py` downloads `predictions.db` from the HF Dataset repo on startup and uploads it back every 30 minutes — platform-agnostic, requires only `HF_TOKEN` and `HF_DATASET_REPO` env vars.

---

## Quick Start

```bash
pip install -r requirements.txt

# Streamlit dashboard
streamlit run main/main_app.py

# FastAPI backend
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

# Classify a headline
python scripts/predict.py -t "India and China sign trade deal" --pair IN-CN

# Live pipeline (single run)
python scripts/live_pipeline.py --once --model both
```

---

## Training Pipeline

```bash
# 1. Collect data
python scripts/collect_corpus.py --feeds 50 --merge
python scripts/fetch_gdelt.py --days 30 --out data/gdelt_raw.csv
python scripts/generate_synthetic.py --backend lmstudio --per-class 200

# 2. Merge (human > GDELT > synthetic) and split
python scripts/merge_augmented_data.py --gdelt-cap 50 --cap-per-class 600
python scripts/split_data.py --input data/labeled_dataset_augmented.csv

# 3. Train
python scripts/train_baseline.py
python scripts/train_transformer.py --epochs 5 --batch-size 32

# 4. Calibrate confidence
python scripts/calibrate_model.py

# 5. Evaluate
python scripts/evaluate.py
python scripts/evaluate_human.py
python scripts/plot_training.py

# 6. Seed predictions DB
python scripts/predict_batch.py --input data/gdelt_raw.csv --model both
```

---

## Project Structure

```
api/            FastAPI serving layer (9 endpoints)
core/           RSS fetching, config, DB sync
data/           Datasets, databases, documentation
main/           Streamlit dashboard
models/         Trained model files
nlp/            Inference, analytics, label mapping, causality
results/        Evaluation outputs and plots
scripts/        Training, evaluation, data pipeline scripts
ui/             Next.js frontend (8 pages)
```

---

## Tech Stack

| Layer | Technologies |
|---|---|
| NLP models | HuggingFace Transformers, scikit-learn, PyTorch |
| API | FastAPI, Uvicorn, Pydantic, APScheduler |
| Dashboard (Streamlit) | Streamlit, Plotly |
| Dashboard (Next.js) | Next.js 16, React 19, Recharts, TypeScript |
| Storage | SQLite, HuggingFace Hub (dataset repo) |
| Data pipeline | feedparser, BeautifulSoup4, GDELT 2.0 |
| Deployment | Docker, Render, Netlify, HF Spaces |
| CI/CD | GitHub Actions |

---

## Author

**Jhanvi Nagori** — Department of Computer Science and Engineering, University of Delhi
