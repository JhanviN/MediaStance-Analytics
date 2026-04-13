#!/usr/bin/env python3
"""
STEP 4b — Fine-tune DistilBERT (3-class) on data/train.csv.

During training, metrics use a **validation** slice (15% of train rows, stratified).
**Test** (`data/test.csv`) is only used once at the end for `transformer_test_predictions.csv`
— so test labels are not used to pick the best epoch.

Requires: pip install datasets accelerate (accelerate often installed with transformers)

Writes:
  models/transformer_bilateral/  (config, tokenizer, pytorch_model.bin or safetensors)
  results/transformer_test_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    logging as hf_logging,
)

hf_logging.set_verbosity_warning()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.label_mapping import ID2LABEL, LABEL2ID, LABELS  # noqa: E402

DATA = ROOT / "data"
MODELS = ROOT / "models" / "transformer_bilateral"
RESULTS = ROOT / "results"

MODEL_NAME = "distilbert-base-uncased"


def _text(row: dict) -> str:
    return (row.get("text") or row.get("headline") or "").strip()


def _load(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=DATA / "train.csv")
    ap.add_argument("--test", type=Path, default=DATA / "test.csv")
    ap.add_argument("--out-dir", type=Path, default=MODELS)
    ap.add_argument("--pred-out", type=Path, default=RESULTS / "transformer_test_predictions.csv")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of train.csv held out for validation (epoch metrics / best checkpoint)",
    )
    args = ap.parse_args()

    train_rows = _load(args.train)
    test_rows = _load(args.test)
    if len(train_rows) < 8 or len(test_rows) < 2:
        sys.exit("Need train.csv and test.csv from split_data.py")

    labels_list = [r["label"].strip().lower() for r in train_rows]
    idx = list(range(len(train_rows)))
    try:
        tr_idx, va_idx = train_test_split(
            idx,
            test_size=args.val_fraction,
            random_state=42,
            stratify=labels_list,
        )
    except ValueError:
        tr_idx, va_idx = train_test_split(
            idx, test_size=args.val_fraction, random_state=42, stratify=None
        )

    fit_rows = [train_rows[i] for i in tr_idx]
    val_rows = [train_rows[i] for i in va_idx]

    ds_train = Dataset.from_dict(
        {
            "text": [_text(r) for r in fit_rows],
            "labels": [LABEL2ID[r["label"].strip().lower()] for r in fit_rows],
        }
    )
    ds_val = Dataset.from_dict(
        {
            "text": [_text(r) for r in val_rows],
            "labels": [LABEL2ID[r["label"].strip().lower()] for r in val_rows],
        }
    )
    ds_test = Dataset.from_dict(
        {
            "text": [_text(r) for r in test_rows],
            "labels": [LABEL2ID[r["label"].strip().lower()] for r in test_rows],
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_len)

    ds_train = ds_train.map(tokenize, batched=True, remove_columns=["text"])
    ds_val = ds_val.map(tokenize, batched=True, remove_columns=["text"])
    ds_test = ds_test.map(tokenize, batched=True, remove_columns=["text"])

    collator = DataCollatorWithPadding(tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float(accuracy_score(labels, preds))}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(args.out_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=20,
        report_to=[],
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    print(
        f"Fit rows: {len(fit_rows)} | Val rows: {len(val_rows)} (for epoch selection) | "
        f"Test rows: {len(test_rows)} (held out until final predict)"
    )
    trainer.save_model(str(args.out_dir))
    tokenizer.save_pretrained(str(args.out_dir))

    # Predictions on test
    pred_out = trainer.predict(ds_test)
    logits = pred_out.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred_ids = np.argmax(logits, axis=-1)

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
                    "pred_label": LABELS[int(pred_ids[i])],
                    "p_adversarial": f"{probs[i, 0]:.6f}",
                    "p_cooperative": f"{probs[i, 1]:.6f}",
                    "p_neutral": f"{probs[i, 2]:.6f}",
                }
            )

    print(f"Saved model -> {args.out_dir}")
    print(f"Saved preds -> {args.pred_out}")


if __name__ == "__main__":
    main()
