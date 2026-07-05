# MediaStance Analytics — Presentation Flow Guide
### For PPT / Slide Deck Construction

---

## SLIDE 1 — Title

**MediaStance Analytics**
*Real-time Bilateral Geopolitical Sentiment Classification from News Headlines*

- Author: Jhanvi N
- Live demo: `https://jhanvin-mediastance-analytics.hf.space`
- GitHub: `github.com/JhanviN/MediaStance-Analytics`

---

## SLIDE 2 — The Problem

**Why does this exist?**

Analysts tracking geopolitical relationships must read hundreds of headlines daily:
- Is India-China trending adversarial this week?
- Did US-Iran tensions spike after the latest sanctions?
- Which headlines are driving the signal?

Generic sentiment analysis (positive/negative) doesn't work here.
"India and China hold border talks" is *positive* in sentiment but *neutral or adversarial* in bilateral stance.

**This is relation-level stance detection** — not sentiment, not NER, not topic modeling.

> 📌 *[Add image: side-by-side of generic sentiment vs bilateral stance on same headline]*

---

## SLIDE 3 — Task Definition

**Input:** A news headline + a country pair
**Output:** One of three labels

| Label | Meaning |
|---|---|
| 🔴 Adversarial | Tension, sanctions, trade war, military action, diplomatic breakdown |
| 🟢 Cooperative | Agreement, trade deal, joint action, diplomatic progress |
| 🔵 Neutral | Ambiguous, factual, no clear bilateral direction |

**Scope:** 15 bilateral pairs across 7 countries
India · China · USA · Russia · Pakistan · Iran · Israel

> 📌 *[Add image: world map with 7 countries highlighted and pair connections drawn]*

---

## SLIDE 4 — System Architecture

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
│  PREDICTIONS DATABASE (SQLite)                       │
│  38,000+ predictions │ 15 pairs │ 2 models           │
└─────────────────────────────────────────────────────┘
              ↓
┌──────────────────┐        ┌──────────────────────────┐
│  FastAPI (REST)  │        │  Next.js Dashboard        │
│  9 endpoints     │ ←────→ │  8 pages                  │
└──────────────────┘        └──────────────────────────┘
              ↑
┌─────────────────────────────────────────────────────┐
│  LIVE PIPELINE (GitHub Actions, daily)               │
│  RSS → predict → DB → HF Dataset sync               │
└─────────────────────────────────────────────────────┘
```

> 📌 *[Use this as a visual flow diagram with colored boxes]*

---

## SLIDE 5 — Data Strategy

**Three-source approach — each source serves a different purpose**

| Source | Rows | Purpose | Noise |
|---|---|---|---|
| Human annotation | 280 | Gold test set only — never trained on | None |
| GDELT (CAMEO codes) | 9,013 enriched | Historical baseline, training variety | ~40% |
| Synthetic (Mistral-7B) | 11,672 | Fill gaps in underrepresented pairs | Low |

**Why GDELT?**
GDELT applies the 30-year CAMEO political science taxonomy to global news. CAMEO codes 1-8 = cooperative, 9-11 = neutral, 12-20 = adversarial. Provides weak supervision at scale — no manual labeling needed.

**Why synthetic?**
GDELT underrepresents India-centric pairs (IN-PK, IN-IR, IL-IN). Synthetic data fills these gaps with pair-specific context using local LLM inference (Mistral-7B via LM Studio).

**Merge priority:** Human > GDELT (capped 50/bucket) > Synthetic

**Final training set:** 11,140 rows — 32.6% adversarial / 34.5% cooperative / 32.9% neutral

> 📌 *[Add image: `results/training_report/label_distribution.png`]*
> 📌 *[Add image: `results/training_report/pair_distribution.png`]*

---

## SLIDE 6 — Key Design Decision: Entity-Aware Input

**The same headline means different things for different pairs.**

> *"India and China hold border talks amid tensions"*
> - For IN-CN: **adversarial** (border tensions)
> - For IN-US: **not relevant**

**Solution:** Prepend the pair to every input:
```
IN-CN: India and China hold border talks amid tensions
```

This teaches the model *which relationship* to evaluate, not just what the headline says.

Without this, a model trained on IN-US data would misclassify IN-CN headlines.

> 📌 *[Add visual: same headline → two different encodings → two different predictions]*

---

## SLIDE 7 — Models

**Baseline: TF-IDF + Logistic Regression**
- 30,000 features, bigrams, sublinear TF weighting
- Balanced class weights
- Fast, interpretable, strong baseline

**Advanced: DistilBERT Fine-tuning**
- Pre-trained: `distilbert-base-uncased` (66M parameters, 40% smaller than BERT)
- Fine-tuned for 5 epochs on Colab T4 GPU (~18 minutes)
- Sequence classification head on [CLS] token
- Optimized for macro F1

**Why DistilBERT over full BERT?**
97% of BERT's performance at 40% fewer parameters — faster inference, fits in free-tier deployment.

> 📌 *[Add image: `results/training_report/loss_curves.png`]*

---

## SLIDE 8 — Confidence Calibration

**Problem:** Neural networks are overconfident. A model saying 94% confidence is often wrong more than 6% of the time.

**Solution: Temperature Scaling**
- Divide logits by temperature T before softmax
- T = 1.5276 (learned on validation set)
- No retraining needed — post-hoc calibration

**Result:**
| Metric | Before | After |
|---|---|---|
| ECE (lower = better) | 0.0647 | 0.0346 |
| Mean confidence | 94.3% | 89.9% |
| Improvement | — | **46% better calibrated** |

**Why this matters:** The `/alerts` endpoint uses confidence thresholds. Uncalibrated confidence makes those thresholds meaningless.

> 📌 *[Add image: `results/training_report/reliability_diagram.png`]*

---

## SLIDE 9 — Evaluation Results

**Two test sets — reported transparently**

| Model | Mixed Test Macro F1 | Human Gold Macro F1 | AUC (avg) |
|---|---|---|---|
| TF-IDF + LR (baseline) | 87.50% | 78.89% | 0.973 |
| DistilBERT (fine-tuned) | **87.84%** | **78.03%** | **0.977** |

**Why two test sets?**
- Mixed (2,785 rows, 87% synthetic): measures task learning
- Human gold (280 rows, 100% human): honest real-world benchmark

The gap between 87.84% and 78.03% is expected — synthetic test data is easier than real headlines.

**Why macro F1?**
Accuracy rewards majority-class prediction. Macro F1 penalizes poor performance on any single class equally — the right metric for a balanced 3-class problem.

> 📌 *[Add image: `results/training_report/confusion_transformer.png`]*
> 📌 *[Add image: `results/training_report/roc_curves.png`]*
> 📌 *[Add image: `results/training_report/per_class_metrics.png`]*
> 📌 *[Add image: `results/training_report/precision_recall_curves.png`]*

---

## SLIDE 10 — Error Analysis

**338 misclassifications analyzed — 3 categories:**

| Category | Share | Example |
|---|---|---|
| Genuine ambiguity | ~40% | "India, China discuss border management" — neutral or adversarial? |
| Data noise | ~30% | GDELT linked wrong article to pair |
| Soft signal detection | ~30% | "Deal still far off", "Strait remains closed" — indirect adversarial |

**Key finding:** The model struggles with *indirect* adversarial language — statements that imply tension without explicit conflict words. This is the hardest subclass and would require more training examples with soft signals.

> 📌 *[Add image: `results/training_report/error_distribution.png`]*
> 📌 *[Add image: `results/error_analysis.md` — paste key examples]*

---

## SLIDE 11 — Attention Visualization

**What does the model actually look at?**

DistilBERT's attention weights extracted from all 6 layers, averaged across 12 heads. CLS token attention shows which words the model focuses on for classification.

**Findings:**
- Adversarial: attends to "sanctions", "tariffs", "tensions", "military", "border"
- Cooperative: attends to "deal", "agreement", "trade", "partnership", "signed"
- Neutral: attends to country names, "talks", "meeting", "discussed"

The model learned semantically correct signals — not just surface keywords.

> 📌 *[Add image: `results/attention/top_words_summary.png`]*
> 📌 *[Add image: any of `results/attention/attention_adversarial_*.png`]*

---

## SLIDE 12 — Live System: Dashboard

**8-page analytics dashboard (Next.js + FastAPI)**

| Page | What it shows |
|---|---|
| Overview | All 15 pairs, color-coded sentiment bars, active alerts |
| Trends | Daily timeline with 7-day rolling average |
| Alerts | Adversarial spike detection (configurable threshold) |
| Headlines | Browse by pair / label / date / confidence |
| Compare Pairs | Side-by-side distribution comparison |
| Live Predict | Classify any headline in real-time |
| Attention | Token attention heatmap |
| Causality | State transition graph, recurring patterns |

> 📌 *[Add screenshots of each page from the live demo]*
> 📌 *[Live demo URL: `https://jhanvin-mediastance-analytics.hf.space`]*

---

## SLIDE 13 — Live System: API

**FastAPI REST API — 9 endpoints**

```
POST /classify              classify any headline
POST /classify/batch        up to 50 headlines at once
GET  /summary/all           all 15 pairs distribution
GET  /trends?pair=CN-US     daily time series
GET  /headlines?pair=IL-IR  browse predictions
GET  /alerts                adversarial spike detection
GET  /compare               side-by-side pair comparison
GET  /causality             state transition graph
GET  /health                health check
```

Auto-generated docs at `/docs` (Swagger UI).

> 📌 *[Add screenshot of `/docs` Swagger UI]*
> 📌 *[Add screenshot of a `/classify` POST request and response]*

---

## SLIDE 14 — Live Pipeline & Deployment

**Fully automated, production-grade pipeline:**

```
GitHub Actions (daily, 6 AM UTC)
    ↓
Download latest DB from HF Dataset repo
    ↓
Fetch fresh headlines (RSS + Google News, last 24h)
    ↓
Run both models (baseline + transformer)
    ↓
Save new predictions to DB
    ↓
Upload updated DB back to HF Dataset repo
    ↓
HF Space downloads updated DB on next sync (every 30 min)
```

**Deployment stack:**
| Component | Platform | URL |
|---|---|---|
| Streamlit dashboard | HF Spaces (free) | `jhanvin-mediastance-analytics.hf.space` |
| FastAPI backend | Render (free) | `mediastance-analytics-1.onrender.com` |
| Next.js frontend | Netlify (free) | *(in progress)* |
| Predictions DB | HF Dataset repo | `JhanviN/mediastance-db` |
| Transformer model | HF Model repo | `JhanviN/mediastance-deploy` |
| Live pipeline | GitHub Actions | Runs daily |

> 📌 *[Add architecture diagram showing all services and data flows]*

---

## SLIDE 15 — Project Stats

| Metric | Value |
|---|---|
| Countries | 7 |
| Bilateral pairs | 15 |
| Training rows | 11,140 |
| Test rows (mixed) | 2,785 |
| Human gold test rows | 280 |
| Predictions in DB | 38,000+ (live, growing) |
| Transformer macro F1 (mixed) | **87.84%** |
| Transformer macro F1 (human gold) | **78.03%** |
| AUC range | 0.964 – 0.982 |
| Calibration ECE improvement | **46%** |
| API endpoints | 9 |
| Dashboard pages | 8 |
| Models trained | 2 |
| Lines of Python | ~4,500 |
| Lines of TypeScript | ~2,000 |

---

## SLIDE 16 — What I Learned / Future Work

**What worked well:**
- Entity-aware input encoding — critical for multi-pair generalization
- Three-source data strategy — each source fills a different gap
- Temperature calibration — makes confidence scores actionable
- Macro F1 as primary metric — honest evaluation

**What's hard:**
- Soft adversarial signals ("deal still far off") — needs more training examples
- GDELT noise — 40% label noise requires careful capping
- Free-tier cold starts — Render spins down after inactivity

**Future work:**
- Inter-annotator agreement (Cohen's kappa with second annotator)
- CAMEO code storage for causal event tracing
- Macro context panel (GDP, political stability indices)
- API authentication and rate limiting

---

## GRAPHS TO INSERT (with file paths)

| Slide | Graph | File |
|---|---|---|
| 5 | Label distribution | `results/training_report/label_distribution.png` |
| 5 | Pair distribution | `results/training_report/pair_distribution.png` |
| 7 | Training loss curves | `results/training_report/loss_curves.png` |
| 8 | Reliability diagram (calibration) | `results/training_report/reliability_diagram.png` |
| 9 | Confusion matrix (transformer) | `results/training_report/confusion_transformer.png` |
| 9 | ROC curves | `results/training_report/roc_curves.png` |
| 9 | Per-class metrics | `results/training_report/per_class_metrics.png` |
| 9 | Precision-recall curves | `results/training_report/precision_recall_curves.png` |
| 10 | Error distribution | `results/training_report/error_distribution.png` |
| 11 | Attention top words | `results/attention/top_words_summary.png` |
| 11 | Attention heatmap (adversarial) | `results/attention/attention_adversarial_CNIN_1.png` |
| 12 | Dashboard screenshots | *(take from live demo)* |
| 13 | API docs screenshot | *(take from `/docs`)* |

---

## SCREENSHOTS TO TAKE FROM LIVE DEMO

1. Overview page — all 15 pairs with color bars
2. Trends page — CN-US line chart with rolling average
3. Alerts page — any active alert
4. Headlines page — adversarial headlines for CN-US
5. Live Predict — type a headline, show result
6. Attention page — heatmap for an adversarial headline
7. Causality page — state transition graph
8. API `/docs` — Swagger UI
9. API `/classify` — POST request + response JSON
