# MediaStance Analytics
## Bilateral Geopolitical Sentiment Classification from News Headlines
### NLP Research Report

---

## Abstract

This paper presents MediaStance Analytics, a supervised NLP system for classifying bilateral geopolitical relationships from news headlines. Given a news headline and a country pair (e.g., India–China), the system classifies the bilateral stance as **adversarial**, **cooperative**, or **neutral**. The system covers 15 bilateral pairs across 7 countries and achieves a macro F1 of 87.84% on a mixed test set and 78.03% on a purely human-annotated gold test set. We employ a three-source data strategy combining human annotation, GDELT event data with CAMEO taxonomy labels, and LLM-generated synthetic augmentation. The final model is a fine-tuned DistilBERT with entity-aware input encoding, temperature-scaled confidence calibration, and a production serving layer with real-time analytics.

---

## 1. Introduction

### 1.1 Problem Statement

Monitoring bilateral geopolitical relationships from news at scale is a critical intelligence task. Analysts tracking India-China trade tensions, US-Iran nuclear negotiations, or Russia-Ukraine sanctions must process hundreds of headlines daily. Manual monitoring is slow, inconsistent, and does not scale.

Existing sentiment analysis systems (positive/negative/neutral) are insufficient for this task. A headline like "India and China hold border talks" is positive in generic sentiment but may be neutral or even adversarial in bilateral stance depending on context. The task requires **relation-level stance detection** — understanding the stance between two specific entities, not just the overall sentiment of the text.

### 1.2 Task Definition

Given:
- A news headline H
- A bilateral pair P = (country_1, country_2)

Predict the stance label L ∈ {adversarial, cooperative, neutral}

**Label definitions:**
- **Adversarial:** The headline signals tension, conflict, sanctions, trade war, military action, diplomatic breakdown, or hostile rhetoric between the two countries
- **Cooperative:** The headline signals agreement, partnership, trade deal, diplomatic progress, joint action, or positive bilateral engagement
- **Neutral:** The headline is ambiguous, reports factual data without clear stance, or discusses both countries without implying a relationship direction

### 1.3 Scope

Countries: India (IN), China (CN), USA (US), Russia (RU), Pakistan (PK), Iran (IR), Israel (IL)

Bilateral pairs (15 total):
CN-IN, CN-IR, CN-PK, CN-RU, CN-US, IL-IN, IL-IR, IL-US, IN-IR, IN-PK, IN-RU, IN-US, IR-RU, IR-US, RU-US

---

## 2. Related Work

### 2.1 Sentiment Analysis vs Stance Detection

Generic sentiment analysis (positive/negative) is well-studied. Bilateral stance detection is a specialized form of **targeted sentiment analysis** where the target is a relationship between two entities rather than a single entity or document.

Prior work on political stance detection (SemEval 2016 Task 6) focused on single-entity stance (e.g., "is this tweet in favor of Hillary Clinton?"). Our task extends this to bilateral relationships — the stance is between two countries, not toward one.

### 2.2 CAMEO Event Taxonomy

The Conflict and Mediation Event Observations (CAMEO) taxonomy (Gerner et al., 2002) provides a structured coding scheme for political events. CAMEO codes 01-08 represent cooperative actions, 09-11 neutral/ambiguous actions, and 12-20 adversarial/conflictual actions. GDELT (Global Database of Events, Language and Tone) applies CAMEO coding to global news at scale, providing a weak supervision signal for our labeling task.

### 2.3 Transfer Learning for NLP

DistilBERT (Sanh et al., 2019) is a distilled version of BERT that retains 97% of BERT's performance at 40% fewer parameters. Fine-tuning pre-trained transformers on domain-specific classification tasks has become standard practice. We follow the approach of Devlin et al. (2019) for sequence classification fine-tuning.

### 2.4 Synthetic Data Augmentation

LLM-generated synthetic data for NLP training has been validated in multiple works including Stanford Alpaca (Taori et al., 2023) and Self-Instruct (Wang et al., 2022). We employ this technique to address class and pair imbalance in our dataset.

---

## 3. Dataset

### 3.1 Data Sources

We employ a three-source data strategy:

**Source 1: Human Annotation (Ground Truth)**
- 280 headlines manually labeled by a single annotator
- Labels: adversarial (131), cooperative (68), neutral (81)
- Used exclusively as gold test set — never in training
- Source: RSS feeds collected over 90 days

**Source 2: GDELT Weak Supervision**
- GDELT 2.0 event files downloaded for 30 days
- 52,381 raw event rows extracted for 15 target pairs
- CAMEO codes mapped to labels: 1-8 → cooperative, 9-11 → neutral, 12-20 → adversarial
- URL enrichment: 9,013 rows enriched with real article titles via parallel HTTP fetching (20 workers)
- Label noise rate: ~40% (CAMEO codes assigned to events, not articles)
- Capped at 50 rows per (pair, label) bucket in training due to noise

**Source 3: Synthetic Augmentation**
- 11,672 headlines generated via Mistral-7B-Instruct (LM Studio, local inference)
- Pair-specific prompts with domain context (e.g., IN-RU: oil imports, S-400, sanctions pressure)
- Three separate generation passes per label class per pair
- Deduplication by content hash
- Balanced across all 15 pairs and 3 labels

**[INSERT TABLE: Data source comparison — rows, label distribution, noise rate]**

### 3.2 Dataset Statistics

After merging with priority ordering (human > GDELT > synthetic):

| Split | Rows | Adversarial | Cooperative | Neutral |
|---|---|---|---|---|
| Train | 11,140 | 3,632 (32.6%) | 3,842 (34.5%) | 3,666 (32.9%) |
| Test (mixed) | 2,785 | 908 (32.6%) | 961 (34.5%) | 916 (32.9%) |
| Human gold | 280 | 131 (46.8%) | 68 (24.3%) | 81 (28.9%) |

**[INSERT FIGURE: Label distribution bar chart — train vs test vs human gold]**

**[INSERT FIGURE: Pair distribution chart — rows per bilateral pair]**

### 3.3 Annotation Guidelines

Labels were assigned according to the following rules:
- **Adversarial:** Explicit signals — tariffs, sanctions, ban, expulsion, military action, retaliation, trade war, diplomatic breakdown. Implicit signals — failed negotiations, closed borders, hostile rhetoric.
- **Cooperative:** Explicit signals — trade deal, agreement, MOU, partnership, summit, joint venture, diplomatic reset. Implicit signals — talks progressing, market access, investment flows.
- **Neutral:** Factual data reports, economic analysis, bilateral meetings scheduled, ongoing negotiations with no clear direction, mixed signals.

**Ambiguity rule:** When a headline contains both cooperative and adversarial signals, label by the dominant signal. When genuinely ambiguous, label neutral.

### 3.4 Data Quality Analysis

**GDELT noise characterization:**
Three types of noise observed in GDELT-enriched data:
1. Wrong article linked to pair (URL fetched different article than GDELT event)
2. Article mentions both countries incidentally (e.g., both attend G20)
3. CAMEO code assigned to sub-event contradicts article's overall stance

**[INSERT FIGURE: GDELT noise examples — 3 cases with headline + CAMEO label + correct label]**

---

## 4. Methodology

### 4.1 Task Framing

We frame bilateral stance detection as a **sequence classification** task with entity-aware input encoding.

**Standard input (baseline):**
```
[CLS] India and China hold trade talks [SEP]
```

**Entity-aware input (our approach):**
```
[CLS] IN-CN: India and China hold trade talks [SEP]
```

The pair prefix `IN-CN:` is prepended to every headline before tokenization. This teaches the model which bilateral relationship to evaluate. Without this, the same headline receives identical encoding regardless of which pair is being queried — a fundamental limitation for multi-pair classification.

**[INSERT FIGURE: Entity-aware encoding diagram — same headline, different pair prefixes, different representations]**

### 4.2 Models

**Model 1: TF-IDF + Logistic Regression (Baseline)**

- Vectorizer: TF-IDF, max 30,000 features, unigrams + bigrams, sublinear TF scaling
- Classifier: Logistic Regression, L2 regularization, balanced class weights, lbfgs solver, max 3,000 iterations
- Input: entity-aware text (pair prefix + headline)
- Training time: ~30 seconds on CPU

**Model 2: DistilBERT Fine-tuning (Advanced)**

- Base model: `distilbert-base-uncased` (66M parameters)
- Classification head: linear layer (768 → 3)
- Input: entity-aware text, max 256 tokens
- Training: 5 epochs, batch size 32, learning rate 2e-5, weight decay 0.01
- Optimization: AdamW with linear warmup
- Early stopping: best checkpoint by validation macro F1
- Hardware: Google Colab T4 GPU, ~18 minutes
- fp16 mixed precision training

**[INSERT FIGURE: DistilBERT architecture diagram with classification head]**

### 4.3 Training Details

**Validation split:** 15% of training data held out for epoch selection (stratified)

**Training objective:** Cross-entropy loss on 3-class classification

**Metric for best checkpoint:** Macro F1 (not accuracy — penalizes poor performance on any single class equally)

**[INSERT FIGURE: Training loss curves — train loss vs validation loss over 5 epochs]**

**[INSERT FIGURE: Validation metrics per epoch — accuracy and macro F1]**

### 4.4 Confidence Calibration

Post-training temperature scaling (Guo et al., 2017) was applied to the transformer model.

**Motivation:** Softmax probabilities from fine-tuned transformers are systematically overconfident. A model outputting 0.94 confidence should be correct 94% of the time — without calibration, this is rarely true.

**Method:** A single scalar temperature T is optimized to minimize negative log-likelihood on the validation set:

```
calibrated_logits = logits / T
calibrated_probs = softmax(calibrated_logits)
```

**Result:**
- Optimal temperature: T = 1.5276 (> 1 confirms overconfidence)
- ECE before: 0.0647
- ECE after: 0.0346 (46% reduction)
- Mean confidence: 0.9432 → 0.8990

**[INSERT FIGURE: Reliability diagram — confidence vs accuracy before and after calibration]**

---

## 5. Experiments and Results

### 5.1 Main Results

**[INSERT TABLE: Full results table]**

| Model | Mixed Accuracy | Mixed Macro F1 | Human Gold Accuracy | Human Gold Macro F1 |
|---|---|---|---|---|
| TF-IDF + LR (Baseline) | 87.54% | 87.50% | 79.64% | 78.89% |
| DistilBERT (Ours) | 87.86% | 87.84% | 78.93% | 78.03% |

### 5.2 Per-Class Results

**Baseline:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Adversarial | 0.8741 | 0.9020 | 0.8878 | 908 |
| Cooperative | 0.8584 | 0.9022 | 0.8798 | 961 |
| Neutral | 0.8974 | 0.8210 | 0.8575 | 916 |
| **Macro avg** | **0.8766** | **0.8750** | **0.8750** | **2785** |

**Transformer:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Adversarial | 0.8804 | 0.8921 | 0.8862 | 908 |
| Cooperative | 0.8919 | 0.8928 | 0.8924 | 961 |
| Neutral | 0.8627 | 0.8504 | 0.8565 | 916 |
| **Macro avg** | **0.8783** | **0.8784** | **0.8784** | **2785** |

**[INSERT FIGURE: Per-class metrics comparison bar chart — precision/recall/F1 for both models]**

### 5.3 ROC-AUC Analysis

One-vs-rest ROC curves computed for each class:

| Class | Baseline AUC | Transformer AUC |
|---|---|---|
| Adversarial | 0.979 | 0.982 |
| Cooperative | 0.977 | 0.981 |
| Neutral | 0.964 | 0.968 |

**[INSERT FIGURE: ROC curves — one vs rest, baseline vs transformer, all 3 classes]**

**[INSERT FIGURE: Precision-Recall curves — one vs rest, baseline vs transformer]**

### 5.4 Confusion Matrix Analysis

**[INSERT FIGURE: Confusion matrix — Baseline (test set)]**

**[INSERT FIGURE: Confusion matrix — Transformer (test set)]**

Key observations:
- Neutral is most frequently confused with cooperative (90 cases baseline, 62 transformer)
- Transformer reduces neutral→cooperative confusion by 31% vs baseline
- Adversarial has highest recall in both models (model is good at detecting hostility)

### 5.5 Human Gold Test Set

Evaluation on 280 purely human-labeled headlines (never seen during training):

| Model | Accuracy | Macro F1 |
|---|---|---|
| Baseline | 79.64% | 78.89% |
| Transformer | 78.93% | 78.03% |

The ~9% gap between mixed test (87.84%) and human gold (78.03%) reflects the distribution shift between synthetic training data and real-world headlines. This is expected and disclosed transparently.

Notably, the baseline slightly outperforms the transformer on human gold data. TF-IDF+LR is less sensitive to distributional shift — it relies on lexical features that generalize more directly. The transformer's advantage is in semantic understanding, which benefits more from larger and more diverse training sets.

### 5.6 Attention Analysis

**[INSERT FIGURE: Top attention words per class — adversarial/cooperative/neutral]**

**[INSERT FIGURE: Sample attention heatmap — adversarial headline with token weights]**

**[INSERT FIGURE: Sample attention heatmap — cooperative headline with token weights]**

Key findings from attention analysis:
- **Adversarial class:** Model attends to "tariffs", "sanctions", "retaliation", "ban", "expel", "restrict"
- **Cooperative class:** Model attends to "agreement", "deal", "partnership", "sign", "cooperation", "invest"
- **Neutral class:** Model attends to "talks", "meeting", "data", "analysis", "review", "ongoing"

This confirms the model learned semantically meaningful features rather than spurious correlations.

---

## 6. Error Analysis

338 misclassifications on the mixed test set (12.14% error rate). Categorized into three types:

### 6.1 Genuine Ambiguity (~40% of errors)

Headlines where even human annotators would disagree:

```
Pair: IR-US | Gold: neutral | Pred: adversarial
"U.S. Imposes Additional Sanctions on Iranian Oil Industry amidst Escalating Tensions"
```

Sanctions are objectively adversarial actions. The human labeled this neutral, the model labeled it adversarial. Both are defensible. This reflects the inherent subjectivity of the neutral class boundary.

### 6.2 Data Noise (~30% of errors)

Headlines where GDELT linked an irrelevant article to a pair:

```
Pair: CN-US | Gold: adversarial | Pred: neutral
"Canada has no intention of pursuing free trade with China, says Carney"
```

This headline is about Canada-China, not CN-US. GDELT tagged it as a CN-US event. The model correctly identifies no CN-US signal — the "error" is in the label, not the prediction.

```
Pair: IN-US | Gold: adversarial | Pred: neutral
"Delhi Assembly receives bomb threat on mail, PM Modi, Amit Shah named"
```

Domestic Indian news with no US bilateral signal. Again, GDELT noise.

### 6.3 Soft Signal Detection (~30% of errors)

Headlines with indirect adversarial signals the model misses:

```
Pair: IR-US | Gold: adversarial | Pred: cooperative
"Iran says final deal still far off as Strait of Hormuz remains closed"
```

"Deal still far off" = failed negotiation = adversarial. "Strait closed" = economic coercion = adversarial. The model sees "deal" and partially associates it with cooperative. This is the model's primary weakness — indirect adversarial language.

**[INSERT FIGURE: Error distribution pie chart — 3 categories]**

---

## 7. Discussion

### 7.1 Baseline vs Transformer

The small performance gap between TF-IDF+LR (87.50%) and DistilBERT (87.84%) on the mixed test set is notable. Several factors explain this:

1. **Data quality:** High-quality synthetic data with clear lexical signals benefits TF-IDF. The model doesn't need deep semantic understanding when training data is clean and consistent.

2. **Domain specificity:** Geopolitical vocabulary is relatively constrained. "Sanctions", "tariffs", "agreement", "deal" are strong lexical signals that TF-IDF captures directly.

3. **Test set composition:** 87% synthetic test data shares the same lexical distribution as training data, benefiting both models equally.

The transformer's advantage is more apparent on the human gold test set in terms of cooperative F1 (0.7973 vs 0.7654) — it better handles the semantic nuance of cooperative language.

### 7.2 Entity-Aware Encoding

The pair prefix (`IN-CN:`) is a simple but effective technique. Without it, the model cannot distinguish which bilateral relationship to evaluate when a headline mentions multiple countries. For example:

```
"India, China, and US discuss trade at G20"
```

With entity-aware encoding:
- `IN-CN: India, China, and US discuss trade at G20` → neutral (India-China context)
- `IN-US: India, China, and US discuss trade at G20` → cooperative (India-US context)
- `CN-US: India, China, and US discuss trade at G20` → neutral (China-US context)

This is analogous to relation extraction approaches where entity markers are injected into the input.

### 7.3 Synthetic Data Ethics and Validity

Using LLM-generated synthetic data raises legitimate questions about evaluation validity. We address this through:

1. **Transparency:** Synthetic data composition is fully disclosed
2. **Separate evaluation:** Human gold test set (280 rows, 0% synthetic) provides an honest benchmark
3. **Standard practice:** Synthetic augmentation is established in NLP (Alpaca, Self-Instruct, etc.)
4. **Distribution diversity:** Three different text sources (human, GDELT, synthetic) prevent the model from overfitting to any single style

### 7.4 Limitations

1. **Single annotator:** Human labels reflect one person's interpretation. Inter-annotator agreement (Cohen's kappa) was not measured — a limitation for academic rigor.

2. **Soft signal weakness:** The model struggles with indirect adversarial signals (failed negotiations, economic pressure without explicit hostile language).

3. **Temporal bias:** Training data covers a specific time period. Geopolitical relationships evolve — a model trained on 2025-2026 data may not generalize to different geopolitical configurations.

4. **English-only:** All sources are English-language. Bilateral relationships involving non-English-dominant countries (Russia, China, Iran) may be underrepresented in English news.

5. **GDELT noise:** ~40% label noise in GDELT-sourced training data. Mitigated by capping at 50 rows per bucket but not eliminated.

---

## 8. System Description

### 8.1 Production Architecture

**[INSERT FIGURE: System architecture diagram]**

```
RSS Feeds (live, every 60 min)
        ↓
Live Pipeline (scripts/live_pipeline.py)
        ↓
DistilBERT / TF-IDF Inference
        ↓
SQLite Predictions DB (18,076 rows)
        ↓
FastAPI REST Layer (9 endpoints)
        ↓
Streamlit Dashboard (7 pages)
```

### 8.2 Real-Time Operation

The system operates continuously via an APScheduler background task embedded in the FastAPI process. Every 60 minutes:
1. Fresh headlines fetched from 6 RSS feeds + 15 Google News pair-specific feeds
2. Entity-aware input constructed for each headline
3. Both models run inference
4. Predictions saved to SQLite with timestamps
5. Dashboard analytics update automatically

### 8.3 Analytics Capabilities

- Label distribution per pair (counts + percentages)
- Daily sentiment time series with configurable rolling average
- Adversarial spike detection (configurable threshold + window)
- Side-by-side pair comparison
- Historical evidence retrieval for predictions
- Attention visualization for model interpretability

---

## 9. Conclusion

We presented MediaStance Analytics, a bilateral geopolitical sentiment classification system achieving 87.84% macro F1 on a mixed test set and 78.03% on a human-annotated gold set. Key contributions:

1. **Task formulation:** Bilateral stance detection as entity-aware sequence classification
2. **Data strategy:** Three-source pipeline (human + GDELT weak supervision + synthetic augmentation) addressing the scarcity of labeled bilateral geopolitical data
3. **Entity-aware encoding:** Pair prefix injection enabling multi-pair classification from a single model
4. **Calibration:** Temperature scaling reducing ECE by 46%, making confidence scores meaningful for alert thresholds
5. **Production system:** End-to-end pipeline from live RSS collection to real-time dashboard analytics

Future work includes inter-annotator agreement validation, causality graph construction from GDELT event sequences, and integration of macroeconomic indicators as contextual features.

---

## References

- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT. NeurIPS Workshop.
- Gerner, D. J., Schrodt, P. A., Yilmaz, O., & Abu-Jabr, R. (2002). Conflict and mediation event observations (CAMEO). ISA Annual Convention.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. ICML.
- Taori, R., et al. (2023). Stanford Alpaca: An instruction-following LLaMA model.
- Wang, Y., et al. (2022). Self-instruct: Aligning language models with self-generated instructions.
- Mohammad, S., et al. (2016). SemEval-2016 Task 6: Detecting stance in tweets. SemEval.

---

## Appendix A — Figures Checklist

Add these figures to the report in the marked locations:

- [ ] Table: Data source comparison (rows, label dist, noise rate)
- [ ] Figure: Label distribution bar chart (train/test/human gold)
- [ ] Figure: Pair distribution chart (rows per pair)
- [ ] Figure: GDELT noise examples (3 cases)
- [ ] Figure: Entity-aware encoding diagram
- [ ] Figure: DistilBERT architecture with classification head
- [ ] Figure: Training loss curves (5 epochs)
- [ ] Figure: Validation metrics per epoch
- [ ] Figure: Reliability diagram (calibration before/after)
- [ ] Figure: Per-class metrics comparison bar chart
- [ ] Figure: ROC curves (3 classes, 2 models)
- [ ] Figure: Precision-Recall curves
- [ ] Figure: Confusion matrix — Baseline
- [ ] Figure: Confusion matrix — Transformer
- [ ] Figure: Top attention words per class
- [ ] Figure: Attention heatmap — adversarial example
- [ ] Figure: Attention heatmap — cooperative example
- [ ] Figure: Error distribution pie chart
- [ ] Figure: System architecture diagram

Most of these are already generated in `results/training_report/`. Run:
```bash
python scripts/plot_training.py
python scripts/attention_viz.py
```

---

## Appendix B — Hyperparameters

| Parameter | Value |
|---|---|
| Base model | distilbert-base-uncased |
| Max sequence length | 256 tokens |
| Learning rate | 2e-5 |
| Batch size | 32 |
| Epochs | 5 |
| Weight decay | 0.01 |
| Warmup | linear |
| Optimizer | AdamW |
| Validation split | 15% |
| Temperature (calibration) | 1.5276 |
| TF-IDF max features | 30,000 |
| TF-IDF n-gram range | (1, 2) |
| LR regularization | L2 (C=1.0) |
| LR solver | lbfgs |
| LR max iterations | 3,000 |
