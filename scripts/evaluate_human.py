"""
Evaluate both models on the human-labeled gold test set only.
This is the honest evaluation — no synthetic data involved.

Usage:
    python scripts/evaluate_human.py
    python scripts/evaluate_human.py --input data/human_gold_test.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.inference import predict_baseline, predict_transformer

DATA = ROOT / "data"
RESULTS = ROOT / "results"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DATA / "human_gold_test.csv")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"Not found: {args.input}\nRun: copy data\\labeled_dataset.csv data\\human_gold_test.csv")

    with open(args.input, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("label", "").strip().lower() in {"adversarial", "cooperative", "neutral"}]

    print(f"Human gold test set: {len(rows)} rows")

    gold, base_preds, trans_preds = [], [], []

    for i, row in enumerate(rows):
        c1 = (row.get("country_1") or "").strip().upper()
        c2 = (row.get("country_2") or "").strip().upper()
        text = (row.get("text") or row.get("headline") or "").strip()
        input_text = f"{c1}-{c2}: {text}" if c1 and c2 else text
        true_label = row["label"].strip().lower()
        gold.append(true_label)

        try:
            lab, _, _ = predict_baseline(input_text)
            base_preds.append(lab)
        except Exception as e:
            base_preds.append("neutral")

        try:
            lab, _, _ = predict_transformer(input_text)
            trans_preds.append(lab)
        except Exception as e:
            trans_preds.append("neutral")

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(rows)}] processed...")

    from sklearn.metrics import classification_report, f1_score, accuracy_score
    labels_order = ["adversarial", "cooperative", "neutral"]

    print("\n" + "="*60)
    print("HUMAN GOLD TEST SET EVALUATION")
    print("="*60)
    print(f"\nTotal human-labeled rows: {len(rows)}")

    print("\n--- Baseline (TF-IDF + Logistic Regression) ---")
    print(f"Accuracy:  {accuracy_score(gold, base_preds)*100:.2f}%")
    print(f"Macro F1:  {f1_score(gold, base_preds, average='macro')*100:.2f}%")
    print(classification_report(gold, base_preds, labels=labels_order, digits=4))

    print("\n--- Transformer (DistilBERT) ---")
    print(f"Accuracy:  {accuracy_score(gold, trans_preds)*100:.2f}%")
    print(f"Macro F1:  {f1_score(gold, trans_preds, average='macro')*100:.2f}%")
    print(classification_report(gold, trans_preds, labels=labels_order, digits=4))

    # Save results
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "human_gold_evaluation.txt"
    lines = [
        "HUMAN GOLD TEST SET — Honest Evaluation (no synthetic data)",
        f"Total rows: {len(rows)}",
        "",
        "Baseline:",
        f"  Accuracy: {accuracy_score(gold, base_preds)*100:.2f}%",
        f"  Macro F1: {f1_score(gold, base_preds, average='macro')*100:.2f}%",
        "",
        "Transformer:",
        f"  Accuracy: {accuracy_score(gold, trans_preds)*100:.2f}%",
        f"  Macro F1: {f1_score(gold, trans_preds, average='macro')*100:.2f}%",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
