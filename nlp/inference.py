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
TEMPERATURE_PATH = TRANSFORMER_DIR / "temperature.json"

_baseline = None
_tokenizer = None
_transformer = None
_temperature: float = 1.0  # default — no scaling


def _load_temperature() -> float:
    """Load calibrated temperature if available, else return 1.0 (no scaling)."""
    global _temperature
    if TEMPERATURE_PATH.exists():
        import json
        data = json.loads(TEMPERATURE_PATH.read_text())
        _temperature = float(data.get("temperature", 1.0))
    return _temperature


def combine_headline_body(headline: str, body: str | None = None, country_1: str = "", country_2: str = "") -> str:
    h = (headline or "").strip()
    b = (body or "").strip()
    # Entity-aware: prepend pair if available
    pair_prefix = ""
    c1 = (country_1 or "").strip().upper()
    c2 = (country_2 or "").strip().upper()
    if c1 and c2:
        pair_prefix = f"{c1}-{c2}: "
    if b and b != h:
        return f"{pair_prefix}{h}. {b}"[:8000]
    return f"{pair_prefix}{h}"


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
    T = _load_temperature()
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**enc).logits
    # Apply temperature scaling — divides logits before softmax
    calibrated_logits = logits / T
    probs_t = torch.softmax(calibrated_logits, dim=-1)[0]
    probs = {LABELS[j]: float(probs_t[j]) for j in range(len(LABELS))}
    idx = int(torch.argmax(probs_t))
    return LABELS[idx], float(probs_t[idx]), probs


def get_attention_weights(text: str, max_length: int = 256) -> Tuple[list[str], list[float], str, float]:
    """
    Returns (tokens, weights, predicted_label, confidence).
    Subword tokens are merged back into words. Weights averaged across all layers/heads.
    """
    tokenizer, model = load_transformer()
    T = _load_temperature()
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
        return_attention_mask=True,
    )
    with torch.no_grad():
        outputs = model(**enc, output_attentions=True)
        logits = outputs.logits
        attentions = outputs.attentions  # tuple of (batch, heads, seq, seq) per layer

    # Prediction
    calibrated_logits = logits / T
    probs_t = torch.softmax(calibrated_logits, dim=-1)[0]
    idx = int(torch.argmax(probs_t))
    label = LABELS[idx]
    confidence = float(probs_t[idx])

    # Average attention across all layers and heads — [CLS] row = what the model attends to
    attn_stack = torch.stack([a[0, :, 0, :] for a in attentions])  # (layers, heads, seq_len)
    avg_attn = attn_stack.mean(dim=(0, 1))  # (seq_len,)

    # Decode tokens and filter padding
    input_ids = enc["input_ids"][0].tolist()
    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    weights_raw = avg_attn.tolist()
    mask = enc["attention_mask"][0].tolist()

    raw_tokens = [t for t, m in zip(raw_tokens, mask) if m == 1]
    weights_raw = [w for w, m in zip(weights_raw, mask) if m == 1]

    # Merge BERT subword tokens (## prefix) back into whole words
    merged_tokens: list[str] = []
    merged_weights: list[float] = []
    for tok, w in zip(raw_tokens, weights_raw):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue  # skip special tokens
        if tok.startswith("##") and merged_tokens:
            merged_tokens[-1] += tok[2:]
            merged_weights[-1] = max(merged_weights[-1], w)  # take max weight for merged token
        else:
            merged_tokens.append(tok)
            merged_weights.append(w)

    # Normalize weights to [0, 1]
    max_w = max(merged_weights) if merged_weights else 1.0
    min_w = min(merged_weights) if merged_weights else 0.0
    rng = max_w - min_w or 1.0
    normalized = [(w - min_w) / rng for w in merged_weights]

    return merged_tokens, normalized, label, confidence
