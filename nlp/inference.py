"""
Load saved baseline + transformer and run inference (same text format as training).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .label_mapping import LABELS

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "models" / "baseline_tfidf_lr.joblib"
TRANSFORMER_DIR = ROOT / "models" / "transformer_bilateral"

_baseline = None
_tokenizer = None
_transformer = None


def combine_headline_body(headline: str, body: str | None = None) -> str:
    h = (headline or "").strip()
    b = (body or "").strip()
    if b and b != h:
        return f"{h}. {b}"[:8000]
    return h


def load_baseline():
    global _baseline
    if _baseline is None:
        if not BASELINE_PATH.exists():
            raise FileNotFoundError(f"Missing baseline model: {BASELINE_PATH} — run train_baseline.py")
        _baseline = joblib.load(BASELINE_PATH)
    return _baseline


def load_transformer():
    global _tokenizer, _transformer
    if _tokenizer is None:
        if not TRANSFORMER_DIR.exists():
            raise FileNotFoundError(
                f"Missing transformer model dir: {TRANSFORMER_DIR} — run train_transformer.py"
            )
        _tokenizer = AutoTokenizer.from_pretrained(str(TRANSFORMER_DIR))
        _transformer = AutoModelForSequenceClassification.from_pretrained(str(TRANSFORMER_DIR))
        _transformer.eval()
    return _tokenizer, _transformer


def predict_baseline(text: str) -> Tuple[str, float, Dict[str, float]]:
    pipe = load_baseline()
    proba = pipe.predict_proba([text])[0]
    idx = int(np.argmax(proba))
    probs = {LABELS[j]: float(proba[j]) for j in range(len(LABELS))}
    return LABELS[idx], float(proba[idx]), probs


def predict_transformer(text: str, max_length: int = 256) -> Tuple[str, float, Dict[str, float]]:
    tokenizer, model = load_transformer()
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**enc).logits
    probs_t = torch.softmax(logits, dim=-1)[0]
    probs = {LABELS[j]: float(probs_t[j]) for j in range(len(LABELS))}
    idx = int(torch.argmax(probs_t))
    return LABELS[idx], float(probs_t[idx]), probs
