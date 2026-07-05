# MediaStance Analytics

**Real-time bilateral geopolitical stance classification from news headlines.**

Given a headline and a country pair, the system classifies the bilateral relationship as `adversarial`, `cooperative`, or `neutral` — covering 15 pairs across 7 countries, updated daily via automated pipeline.

---

## Live Deployments

| Component | URL | Platform |
|---|---|---|
| Next.js Dashboard | [mediastance.netlify.app](https://tourmaline-klepon-02ce4e.netlify.app/) | Netlify |
| FastAPI Backend | [mediastance-analytics-1.onrender.com/docs](https://mediastance-analytics-1.onrender.com/docs) | Render |
| Streamlit Dashboard | [HuggingFace Spaces](https://huggingface.co/spaces/JhanviN/mediastance-analytics) | HF Spaces |

---

## Model Performance

| Model | Mixed Test Macro F1 | Human Gold Macro F1 | Avg AUC |
|---|---|---|---|
| TF-IDF + Logistic Regression | 87.50% | 78.89% | 0.973 |
| DistilBERT (fine-tuned) | **87.84%** | **78.03%** | **0.977** |

The **human gold test set** (280 rows, 100% human-annotated) is the honest benchmark. The mixed test set (2,785 rows, 87% synthetic) measures task learning. Both are reported transparently.

> 38,000+ predictions in DB · 15 bilateral pairs · 7 countries · Temperature-calibrated confidence (ECE −46%)

---

## The Problem This Solves

Tracking geopolitical relationships from news requires reading hundreds of headlines daily across dozens of pairs. Existing sentiment tools assign generic positive/negative scores — they don't capture *bilateral stance* (the same headline about "trade talks" can be cooperative for one pair and adversarial for another, depending on context).

MediaStance frames this as a 3-class relation extraction problem with entity-aware encoding, and wraps the predictions into a live analytics system that surfaces trend shifts, adversarial spikes, and state transition patterns — without a human reading every article.

---

## Countries & Pairs

**7 countries:** India · China · USA · Russia · Pakistan · Iran · Israel

**15 bilateral pairs:** CN-IN · CN-IR · CN-PK · CN-RU · CN-US · IL-IN · IL-IR · IL-US · IN-IR · IN-PK · IN-RU · IN-US · IR-RU · IR-US · RU-US

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  DATA SOURCES                                         │
│  RSS / Google News (live)  ·  GDELT (historical)     │
│  Human annotation (gold labels)  ·  Synthetic (LLM)  │
└───────────────────────┬──────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│  NLP MODELS                                           │
│  TF-IDF + Logistic Regression (baseline)             │
│  DistilBERT fine-tuned (entity-aware, temp-scaled)   │
└───────────────────────┬──────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│  PREDICTIONS DATABASE  (SQLite, 38k+ rows)            │
│  Persisted in HF Dataset repo · synced every 30 min  │
└───────────────────────┬──────────────────────────────┘
                        ↓
          ┌─────────────┴──────────────┐
          ↓                            ↓
  ┌───────────────┐          ┌────────────────────┐
  │  FastAPI      │          │  Streamlit          │
  │  9 endpoints  │ ◄──────► │  + Next.js (8 pages)│
  └───────────────┘          └────────────────────┘
          ↑
┌──────────────────────────────────────────────────────┐
│  LIVE PIPELINE  (GitHub Actions · daily 06:00 UTC)    │
│  RSS → predict (both models) → DB → HF Dataset sync  │
└──────────────────────────────────────────────────────┘
```

---

## Dashboard Pages

| Page | Description |
|---|---|
| Overview | All 15 pairs with color-coded sentiment bars and active alerts |
| Trends | Daily time series with 7-day rolling average and date filters |
| Alerts | Adversarial spike detection with configurable threshold and window |
| Headlines | Browse predictions by pair, label, date, and confidence |
| Compare Pairs | Side-by-side distribution comparison of any two pairs |
| Live Predict | Real-time classification of any headline with historical context |
| Attention | Token attention heatmap — model interpretability via DistilBERT CLS attention |
| Causality | State transition graph, spike analysis, and recurring 3-step patterns |

---

## API Endpoints

```
POST /classify              — classify a single headline
POST /classify/batch        — up to 50 headlines per request
GET  /summary/all           — label distribution across all 15 pairs
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
Every input is prefixed with the bilateral pair: `IN-CN: India and China hold border talks`. This forces the model to evaluate the same headline from the perspective of a specific relationship — without it, identical headlines receive identical encodings regardless of which pair is being queried.

**Three-source training strategy with priority ordering**
Human annotation → GDELT weak supervision (capped at 50 rows/bucket, ~40% noise) → Synthetic augmentation (fills underrepresented pairs). Implemented in `training/merge_augmented_data.py`. Result: 13,925 rows, near-perfect label balance (33%/34%/33%).

**Two honest test sets**
Mixed test set (2,785 rows, 87% synthetic) for task learning. Human gold set (280 rows, 100% human-labeled) for real-world generalization. Both published.

**Temperature scaling**
Post-hoc confidence calibration (T = 1.5276). ECE reduced from 0.0647 → 0.0346 (46% improvement). Makes confidence thresholds in `/alerts` meaningful rather than overconfident softmax outputs.

**Macro F1 as primary metric**
Penalizes poor performance on any individual class equally — correct for a balanced 3-class problem. Accuracy would reward majority-class exploitation.

**Ephemeral container storage pattern**
Both HF Spaces and Render use containers that reset on restart. `core/db_sync.py` treats a private HF Dataset repo as the canonical store — downloads `predictions.db` on startup, uploads every 30 minutes, and registers a final upload on shutdown via `atexit`. Requires only `HF_TOKEN` and `HF_DATASET_REPO` env vars.

---

## Quick Start

```bash
git clone https://github.com/JhanviN/MediaStance-Analytics
cd MediaStance-Analytics
pip install -r requirements.txt

# Streamlit dashboard
streamlit run main/main_app.py

# FastAPI backend (with auto-scheduler)
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

# CLI: classify a single headline
python pipeline/predict.py -t "India and China sign trade deal" --pair IN-CN

# Live pipeline: single run
python pipeline/live_pipeline.py --once --model both
```

---

## Training Pipeline

The full training sequence is one-time. Artifacts (`models/`) are committed.

```bash
# 1. Collect training data
python training/collect_corpus.py --feeds 50 --merge
python training/fetch_gdelt.py --days 30 --out data/gdelt_raw.csv
python training/generate_synthetic.py --backend lmstudio --per-class 200

# 2. Merge and split
python training/merge_augmented_data.py --gdelt-cap 50 --cap-per-class 600
python training/split_data.py --input data/labeled_dataset_augmented.csv

# 3. Train
python training/train_baseline.py
python training/train_transformer.py --epochs 5 --batch-size 32

# 4. Calibrate confidence
python training/calibrate_model.py

# 5. Evaluate
python training/evaluate.py
python training/evaluate_human.py

# 6. Seed predictions DB from GDELT history
python pipeline/predict_batch.py --input data/gdelt_raw.csv --model both
```

---

## Project Structure

```
api/              FastAPI serving layer — 9 endpoints, APScheduler, DB sync on startup
core/             Config, RSS fetching, HF Dataset DB sync
data/             Datasets and SQLite predictions database
docs/             Project documentation, research report, presentation notes
main/             Streamlit dashboard (8 pages)
models/           Trained model artifacts (baseline .joblib + DistilBERT weights)
nlp/              Inference, analytics queries, causality, label mapping, attention viz
pipeline/         Live pipeline, batch predict, CLI predict, weekly report
results/          Evaluation outputs, confusion matrices, attention plots, ROC curves
training/         One-time training, evaluation, and data pipeline scripts
ui/               Next.js frontend (8 pages, TypeScript, Recharts)
```

---

## Deployment

| Component | Platform | Config |
|---|---|---|
| Streamlit dashboard | HF Spaces (Docker) | `Dockerfile`, port 7860 |
| FastAPI backend | Render (Docker) | `Dockerfile.api`, `render.yaml` |
| Next.js frontend | Netlify | `ui/netlify.toml` |
| Predictions DB | HF Dataset repo | `JhanviN/mediastance-db` |
| Transformer model | HF Model repo | `JhanviN/mediastance-deploy` |
| Live pipeline | GitHub Actions | Daily at 06:00 UTC |

---

## Tech Stack

| Layer | Technologies |
|---|---|
| NLP / ML | HuggingFace Transformers, PyTorch, scikit-learn |
| API | FastAPI, Uvicorn, Pydantic, APScheduler |
| Streamlit dashboard | Streamlit, Plotly, Matplotlib |
| Next.js dashboard | Next.js 16, React 19, TypeScript, Recharts |
| Storage | SQLite, HuggingFace Hub (dataset + model repos) |
| Data pipeline | feedparser, BeautifulSoup4, GDELT 2.0 |
| Deployment | Docker, Render, Netlify, HF Spaces, GitHub Actions |

---

## Author

**Jhanvi Nagori** · Department of Computer Science and Engineering, University of Delhi
