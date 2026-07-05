
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _load(path: Path) -> tuple[list[str], list[str]]:
    gold, pred = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            gold.append(row["true_label"].strip().lower())
            pred.append(row["pred_label"].strip().lower())
    return gold, pred


def _md_block(title: str, gold: list[str], pred: list[str], labels: list[str]) -> str:
    acc = accuracy_score(gold, pred)
    rep = classification_report(
        gold, pred, labels=labels, digits=4, zero_division=0
    )
    cm = confusion_matrix(gold, pred, labels=labels)
    lines = [
        f"## {title}",
        "",
        f"**Accuracy:** {acc:.4f}",
        "",
        "### Classification report",
        "",
        "```",
        rep.rstrip(),
        "```",
        "",
        "### Confusion matrix (rows=true, cols=pred)",
        "",
        "| | " + " | ".join(labels) + " |",
        "|---|" + "|".join(["---"] * len(labels)) + "|",
    ]
    for i, lab in enumerate(labels):
        row = [lab] + [str(cm[i, j]) for j in range(len(labels))]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-pred", type=Path, default=RESULTS / "baseline_test_predictions.csv")
    ap.add_argument("--transformer-pred", type=Path, default=RESULTS / "transformer_test_predictions.csv")
    ap.add_argument("--out", type=Path, default=RESULTS / "evaluation_report.md")
    args = ap.parse_args()

    labels = ["adversarial", "cooperative", "neutral"]
    parts = ["# Bilateral sentiment — test set evaluation\n"]

    if args.baseline_pred.exists():
        g, p = _load(args.baseline_pred)
        parts.append(_md_block("Baseline (TF-IDF + Logistic Regression)", g, p, labels))
    else:
        parts.append("## Baseline\n\n*(missing baseline predictions — run train_baseline.py)*\n")

    if args.transformer_pred.exists():
        g, p = _load(args.transformer_pred)
        parts.append(_md_block("Advanced (fine-tuned DistilBERT)", g, p, labels))
    else:
        parts.append("## Advanced\n\n*(missing transformer predictions — run train_transformer.py)*\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
