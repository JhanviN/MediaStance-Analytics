---
title: MediaStance Analytics
emoji: 📡
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: 1.33.0
app_file: main/main_app.py
pinned: true
license: mit
---

# MediaStance Analytics

Real-time bilateral geopolitical sentiment classification from news headlines.

**Macro F1: 87.84%** | 15 country pairs | 7 countries | Live dashboard

## What it does

Given a news headline and a country pair (e.g., India-China), classifies the bilateral relationship as **adversarial**, **cooperative**, or **neutral** using a fine-tuned DistilBERT model.

## Countries & Pairs

India, China, USA, Russia, Pakistan, Iran, Israel — 15 bilateral pairs

## Dashboard Pages

- **Overview** — All 15 pairs with real-time sentiment percentages
- **Trends** — Daily sentiment timeline with rolling average
- **Alerts** — Adversarial spike detection
- **Headlines** — Browse predictions by pair and label
- **Compare Pairs** — Side-by-side comparison
- **Live Predict** — Classify any headline with historical context
- **Attention** — Token attention heatmap (model interpretability)
- **Causality** — State transition graph over time

## Model

- Baseline: TF-IDF + Logistic Regression (87.50% macro F1)
- Advanced: Fine-tuned DistilBERT (87.84% macro F1)
- Confidence calibration: Temperature scaling (ECE improved 46%)

## GitHub

[MediaStance Analytics](https://github.com/JhanviN/MediaStance-Analytics)
