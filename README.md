# MediaStance Analytics

**Real-time bilateral geopolitical sentiment classification from news headlines.**

Given a headline and a country pair, the system classifies the bilateral relationship as `adversarial`, `cooperative`, or `neutral`.

---

## Results

| Model | Mixed Test Macro F1 | Human Gold Macro F1 | AUC (avg) |
|---|---|---|---|
| TF-IDF + Logistic Regression | 87.50% | 78.89% | 0.973 |
| DistilBERT (fine-tuned) | **87.84%** | **78.03%** | **0.977** |

15 bilateral pairs · 7 countries · 18,076 predictions in DB · Temperature-calibrated confidence

---

## Countries & Pairs

**Countries:** India (IN), China (CN), USA (US), Russia (RU), Pakistan (PK), Iran (IR), Israel (IL)

**Pairs (15):** CN-IN, CN-IR, CN-PK, CN-RU, CN-US, IL-IN, IL-IR, IL-US, IN-IR, IN-PK, IN-RU, IN-US, IR-RU, IR-US, RU-US

---

## Quick Start

```bash
pip install -r requirements.txt

# Dashboard
streamlit run demo/demo_app.py

# API (auto-runs live pipeline every 60 min)
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

# Classify a headline
python scripts/predict.py -t "India and China sign trade deal" --pair IN-CN

# Live pipeline (manual run)
python scripts/live_pipeline.py --once --model baseline
```

---

## Architecture

```
RSS Feeds (live, every 60 min)
        ↓
Live Pipeline → DistilBERT / TF-IDF
        ↓
SQLite Predictions DB
        ↓
FastAPI (9 endpoints) + Streamlit Dashboard (7 pages)
```

---

## Dashboard Pages

| Page | Description |
|---|---|
| 🏠 Overview | All 15 pairs, color-coded sentiment, active alerts |
| 📈 Trends | Daily timeline with rolling average |
| 🔴 Alerts | Adversarial spike detection |
| 📰 Headlines | Browse by pair, label, confidence |
| ⚖️ Compare Pairs | Side-by-side comparison |
| 🔮 Live Predict | Classify any headline + historical context |
| 🧠 Attention | Token attention heatmap (model interpretability) |

---

## API Endpoints

```
POST /classify              — single headline
POST /classify/batch        — up to 50 headlines
GET  /summary?pair=IN-US    — label distribution
GET  /summary/all           — all 15 pairs
GET  /trends?pair=CN-US     — daily time series
GET  /headlines?pair=IL-IR  — browse predictions
GET  /alerts                — adversarial spike detection
GET  /compare?pair1=IN-US&pair2=CN-US
GET  /health
```

Docs: `http://127.0.0.1:8000/docs`

---

## Training Pipeline

```bash
# 1. Collect data
python scripts/collect_corpus.py --feeds 50 --merge
python scripts/fetch_gdelt.py --days 30 --out data/gdelt_raw.csv
python scripts/generate_synthetic.py --backend lmstudio --per-class 200 --out data/synthetic_raw.csv

# 2. Merge and split
python scripts/merge_augmented_data.py --gdelt-cap 50 --cap-per-class 9999
python scripts/split_data.py --input data/labeled_dataset_augmented.csv

# 3. Train
python scripts/train_baseline.py
python scripts/train_transformer.py --epochs 5 --batch-size 16

# 4. Calibrate confidence
python scripts/calibrate_model.py

# 5. Evaluate
python scripts/evaluate.py
python scripts/evaluate_human.py
python scripts/plot_training.py

# 6. Populate DB with historical data
python scripts/predict_batch.py --input data/gdelt_raw.csv --model baseline
```

---

## Key Technical Details

- **Entity-aware input:** Pair prepended to headline (`IN-CN: headline text`) so the model knows which relationship to evaluate
- **Macro F1:** Used instead of accuracy — penalizes poor performance on any single class equally
- **Temperature scaling:** Calibrated confidence (T=1.5276, ECE improved 46%)
- **Two test sets:** Mixed (87% synthetic) + human gold (100% human-labeled) for honest evaluation
- **APScheduler:** Live pipeline runs automatically inside the FastAPI process — no cron jobs needed

---

## Project Structure

```
api/                    FastAPI serving layer
core/                   RSS fetching, config
data/                   Datasets and databases
demo/                   Streamlit dashboard
models/                 Trained model files
nlp/                    Inference, analytics, label mapping
results/                Evaluation outputs and plots
scripts/                Training, evaluation, data pipeline scripts
```

---

## Documentation

- `PROJECT_DOC.md` — complete implementation documentation
- `RESEARCH_REPORT.md` — NLP research report with methodology and results
  

Author  
Jhanvi Nagori  
Fullstack Developer
