

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.label_mapping import ID2LABEL, LABEL2ID, LABELS  # noqa: E402

DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def _text(row: dict) -> str:
    return (row.get("text") or row.get("headline") or "").strip()


def _load(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=DATA / "train.csv")
    ap.add_argument("--test", type=Path, default=DATA / "test.csv")
    ap.add_argument("--model-out", type=Path, default=MODELS / "baseline_tfidf_lr.joblib")
    ap.add_argument("--pred-out", type=Path, default=RESULTS / "baseline_test_predictions.csv")
    args = ap.parse_args()

    train_rows = _load(args.train)
    test_rows = _load(args.test)
    if not train_rows or not test_rows:
        sys.exit("train.csv and test.csv must exist. Run: python scripts/split_data.py")

    X_train = [_text(r) for r in train_rows]
    y_train = [LABEL2ID[r["label"].strip().lower()] for r in train_rows]
    X_test = [_text(r) for r in test_rows]
    y_test = [LABEL2ID[r["label"].strip().lower()] for r in test_rows]

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=30_000,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.model_out)
    mapping_path = MODELS / "label_mapping.json"
    mapping_path.write_text(
        json.dumps({"labels": LABELS, "label2id": LABEL2ID, "id2label": {str(k): v for k, v in ID2LABEL.items()}}),
        encoding="utf-8",
    )

    proba = pipe.predict_proba(X_test)
    pred = np.argmax(proba, axis=1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "true_label",
        "pred_label",
        "p_adversarial",
        "p_cooperative",
        "p_neutral",
    ]
    with open(args.pred_out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, row in enumerate(test_rows):
            w.writerow(
                {
                    "id": row.get("id", ""),
                    "true_label": row["label"],
                    "pred_label": LABELS[int(pred[i])],
                    "p_adversarial": f"{proba[i, 0]:.6f}",
                    "p_cooperative": f"{proba[i, 1]:.6f}",
                    "p_neutral": f"{proba[i, 2]:.6f}",
                }
            )

    acc = (pred == np.array(y_test)).mean()
    print(f"Train accuracy (in-sample, not for report): {pipe.score(X_train, y_train):.4f}")
    print(f"Test accuracy (hold-out): {acc:.4f}")
    print(f"Saved model -> {args.model_out}")
    print(f"Saved preds -> {args.pred_out}")


if __name__ == "__main__":
    main()
