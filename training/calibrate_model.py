"""
Temperature scaling calibration for DistilBERT.

After fine-tuning, softmax probabilities are often overconfident.
Temperature scaling finds a single scalar T that divides logits before softmax,
making confidence scores match actual accuracy (calibrated probabilities).

Usage:
    python scripts/calibrate_model.py
    python scripts/calibrate_model.py --val data/test.csv

After running, the temperature is saved to models/transformer_bilateral/temperature.json
and automatically used by nlp/inference.py for all future predictions.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.label_mapping import LABEL2ID, LABELS
from nlp.inference import load_transformer

DATA = ROOT / "data"
MODELS = ROOT / "models" / "transformer_bilateral"
TEMP_PATH = MODELS / "temperature.json"


def _load_val(path: Path) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            lab = row.get("label", "").strip().lower()
            if lab not in LABEL2ID:
                continue
            c1 = (row.get("country_1") or "").strip().upper()
            c2 = (row.get("country_2") or "").strip().upper()
            text = (row.get("text") or row.get("headline") or "").strip()
            input_text = f"{c1}-{c2}: {text}" if c1 and c2 else text
            texts.append(input_text)
            labels.append(LABEL2ID[lab])
    return texts, labels


def get_logits(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Run model forward pass and collect raw logits (before softmax)."""
    tokenizer, model = load_transformer()
    model.eval()
    all_logits = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=256, padding=True,
        )
        with torch.no_grad():
            logits = model(**enc).logits
        all_logits.append(logits.cpu().numpy())
        if (i // batch_size + 1) % 5 == 0:
            print(f"  [{i + batch_size}/{len(texts)}] logits collected...")

    return np.vstack(all_logits)


class TemperatureScaler(nn.Module):
    """Single scalar temperature applied to logits before softmax."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def find_temperature(logits: np.ndarray, labels: list[int]) -> float:
    """
    Optimize temperature T to minimize NLL on validation set.
    Uses L-BFGS optimizer — converges in <100 iterations.
    """
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)

    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=100)
    criterion = nn.CrossEntropyLoss()

    def eval_step():
        optimizer.zero_grad()
        scaled = scaler(logits_t)
        loss = criterion(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(eval_step)
    return float(scaler.temperature.item())


def expected_calibration_error(probs: np.ndarray, labels: list[int], n_bins: int = 10) -> float:
    """ECE — measures how well confidence matches accuracy across probability bins."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == np.array(labels)).astype(float)

    ece = 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(ece / len(labels))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Temperature scaling calibration")
    ap.add_argument("--val", type=Path, default=DATA / "test.csv",
                    help="Validation CSV (default: data/test.csv)")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    if not args.val.exists():
        sys.exit(f"Validation file not found: {args.val}")

    print(f"Loading validation data from {args.val.name}...")
    texts, labels = _load_val(args.val)
    print(f"  {len(texts)} validation examples")

    print("\nCollecting logits from transformer model...")
    logits = get_logits(texts, batch_size=args.batch_size)

    # Before calibration
    probs_before = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    ece_before = expected_calibration_error(probs_before, labels)
    acc_before = (probs_before.argmax(axis=1) == np.array(labels)).mean()
    print(f"\nBefore calibration:")
    print(f"  Accuracy: {acc_before*100:.2f}%")
    print(f"  ECE (lower=better): {ece_before:.4f}")
    print(f"  Mean confidence: {probs_before.max(axis=1).mean():.4f}")

    # Find optimal temperature
    print("\nOptimizing temperature...")
    T = find_temperature(logits, labels)
    print(f"  Optimal temperature: {T:.4f}")

    # After calibration
    probs_after = torch.softmax(torch.tensor(logits) / T, dim=-1).numpy()
    ece_after = expected_calibration_error(probs_after, labels)
    print(f"\nAfter calibration (T={T:.4f}):")
    print(f"  Accuracy: {acc_before*100:.2f}%  (unchanged — calibration doesn't affect predictions)")
    print(f"  ECE (lower=better): {ece_after:.4f}  (was {ece_before:.4f})")
    print(f"  Mean confidence: {probs_after.max(axis=1).mean():.4f}")
    print(f"  ECE improvement: {(ece_before - ece_after)*100:.1f}%")

    # Save temperature
    TEMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEMP_PATH.write_text(json.dumps({"temperature": T, "ece_before": ece_before, "ece_after": ece_after}))
    print(f"\nSaved temperature → {TEMP_PATH}")
    print("inference.py will automatically use this temperature for all future predictions.")


if __name__ == "__main__":
    main()
