"""
Generate comprehensive training visualization from trainer_state.json.
Run after training completes:
    python scripts/plot_training.py

Produces results/training_report/ with:
  - loss_curves.png        — train + eval loss per step
  - accuracy_f1.png        — accuracy + macro F1 per epoch
  - confusion_matrix.png   — on test set
  - per_class_metrics.png  — precision, recall, F1 per class
  - training_summary.txt   — key numbers for presentation
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS = ROOT / "models" / "transformer_bilateral"
RESULTS = ROOT / "results"
REPORT_DIR = RESULTS / "training_report"
DATA = ROOT / "data"


def _load_trainer_state() -> dict:
    path = MODELS / "trainer_state.json"
    if not path.exists():
        # Try checkpoints
        for cp in sorted(MODELS.glob("checkpoint-*")):
            p = cp / "trainer_state.json"
            if p.exists():
                path = p
                break
    if not path.exists():
        sys.exit(f"trainer_state.json not found in {MODELS}. Run training first.")
    return json.loads(path.read_text())


def plot_loss_curves(state: dict, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("  [skip] matplotlib not installed — run: pip install matplotlib")
        return

    log = state.get("log_history", [])
    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []
    eval_acc, eval_f1 = [], []

    for entry in log:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_loss.append(entry["eval_loss"])
            if "eval_accuracy" in entry:
                eval_acc.append(entry["eval_accuracy"])
            if "eval_macro_f1" in entry:
                eval_f1.append(entry["eval_macro_f1"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("TradePulse — DistilBERT Training", fontsize=14, fontweight="bold")

    # Loss curves
    ax = axes[0]
    if train_steps:
        ax.plot(train_steps, train_loss, label="Train Loss", color="#2196F3", alpha=0.8, linewidth=1.5)
    if eval_steps:
        ax.plot(eval_steps, eval_loss, label="Val Loss", color="#F44336", linewidth=2.5, marker="o", markersize=6)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy + F1
    ax = axes[1]
    epochs = list(range(1, len(eval_acc) + 1))
    if eval_acc:
        ax.plot(epochs, [a * 100 for a in eval_acc], label="Val Accuracy %", color="#4CAF50", linewidth=2.5, marker="o", markersize=8)
    if eval_f1:
        ax.plot(epochs, [f * 100 for f in eval_f1], label="Val Macro F1 %", color="#FF9800", linewidth=2.5, marker="s", markersize=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score (%)")
    ax.set_title("Validation Metrics per Epoch")
    ax.set_xticks(epochs)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    out = out_dir / "loss_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


def plot_confusion_matrix(pred_path: Path, out_dir: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        return

    if not pred_path.exists():
        print(f"  [skip] {pred_path.name} not found")
        return

    from sklearn.metrics import confusion_matrix, classification_report
    labels_order = ["adversarial", "cooperative", "neutral"]

    gold, pred = [], []
    with open(pred_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            gold.append(row["true_label"].strip().lower())
            pred.append(row["pred_label"].strip().lower())

    cm = confusion_matrix(gold, pred, labels=labels_order)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="% of true class")

    ax.set_xticks(range(len(labels_order)))
    ax.set_yticks(range(len(labels_order)))
    ax.set_xticklabels([l.capitalize() for l in labels_order])
    ax.set_yticklabels([l.capitalize() for l in labels_order])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {title}")

    for i in range(len(labels_order)):
        for j in range(len(labels_order)):
            color = "white" if cm_pct[i, j] > 60 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=10, color=color, fontweight="bold")

    plt.tight_layout()
    fname = f"confusion_{title.lower().replace(' ', '_')}.png"
    out = out_dir / fname
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")

    # Print classification report
    print(f"\n  {title} Classification Report:")
    print(classification_report(gold, pred, labels=labels_order, digits=4))


def plot_per_class_metrics(pred_paths: dict, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        return

    from sklearn.metrics import precision_recall_fscore_support
    labels_order = ["adversarial", "cooperative", "neutral"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Per-Class Metrics Comparison", fontsize=13, fontweight="bold")
    metrics_names = ["Precision", "Recall", "F1-Score"]
    colors = {"Baseline": "#2196F3", "Transformer": "#4CAF50"}

    all_data = {}
    for model_name, pred_path in pred_paths.items():
        if not pred_path.exists():
            continue
        gold, pred = [], []
        with open(pred_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                gold.append(row["true_label"].strip().lower())
                pred.append(row["pred_label"].strip().lower())
        p, r, f, _ = precision_recall_fscore_support(gold, pred, labels=labels_order, zero_division=0)
        all_data[model_name] = {"Precision": p, "Recall": r, "F1-Score": f}

    x = np.arange(len(labels_order))
    width = 0.35

    for ax_idx, metric in enumerate(metrics_names):
        ax = axes[ax_idx]
        for i, (model_name, data) in enumerate(all_data.items()):
            offset = (i - len(all_data) / 2 + 0.5) * width
            bars = ax.bar(x + offset, data[metric], width,
                         label=model_name, color=colors.get(model_name, "#9C27B0"), alpha=0.85)
            for bar, val in zip(bars, data[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels([l.capitalize() for l in labels_order])
        ax.set_ylim(0, 1.15)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = out_dir / "per_class_metrics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


def write_summary(state: dict, pred_paths: dict, out_dir: Path) -> None:
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    labels_order = ["adversarial", "cooperative", "neutral"]

    lines = [
        "=" * 60,
        "TradePulse — Training Summary",
        "=" * 60,
        "",
    ]

    # Best epoch from trainer state
    log = state.get("log_history", [])
    best_f1 = max((e.get("eval_macro_f1", 0) for e in log if "eval_macro_f1" in e), default=0)
    best_acc = max((e.get("eval_accuracy", 0) for e in log if "eval_accuracy" in e), default=0)
    total_steps = state.get("global_step", "?")
    best_epoch = state.get("best_model_checkpoint", "?")

    lines += [
        f"Total training steps:  {total_steps}",
        f"Best checkpoint:       {best_epoch}",
        f"Best val accuracy:     {best_acc*100:.2f}%",
        f"Best val macro F1:     {best_f1*100:.2f}%",
        "",
    ]

    for model_name, pred_path in pred_paths.items():
        if not pred_path.exists():
            continue
        gold, pred = [], []
        with open(pred_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                gold.append(row["true_label"].strip().lower())
                pred.append(row["pred_label"].strip().lower())
        acc = accuracy_score(gold, pred)
        f1 = f1_score(gold, pred, average="macro")
        lines += [
            f"--- {model_name} (test set) ---",
            f"  Accuracy:   {acc*100:.2f}%",
            f"  Macro F1:   {f1*100:.2f}%",
            "",
            classification_report(gold, pred, labels=labels_order, digits=4),
        ]

    out = out_dir / "training_summary.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved {out.name}")
    print("\n" + "\n".join(lines[:12]))


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating training report...")

    state = _load_trainer_state()
    pred_paths = {
        "Baseline": RESULTS / "baseline_test_predictions.csv",
        "Transformer": RESULTS / "transformer_test_predictions.csv",
    }

    print("\n[1/4] Loss curves + validation metrics")
    plot_loss_curves(state, REPORT_DIR)

    print("\n[2/4] Confusion matrices")
    plot_confusion_matrix(pred_paths["Baseline"], REPORT_DIR, "Baseline")
    plot_confusion_matrix(pred_paths["Transformer"], REPORT_DIR, "Transformer")

    print("\n[3/4] Per-class metrics comparison")
    plot_per_class_metrics(pred_paths, REPORT_DIR)

    print("\n[4/4] ROC curves (AUC)")
    plot_roc_curves(pred_paths, REPORT_DIR)

    print("\n[5/5] Precision-Recall curves")
    plot_precision_recall_curves(pred_paths, REPORT_DIR)

    print("\n[6/6] Training summary")
    write_summary(state, pred_paths, REPORT_DIR)

    print(f"\nAll plots saved to {REPORT_DIR}")
    print("Open results/training_report/ to review before deciding on retraining.")


def plot_roc_curves(pred_paths: dict, out_dir: Path) -> None:
    """One-vs-rest ROC curves for each class, both models on same plot."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        return

    from sklearn.metrics import roc_curve, auc
    import numpy as np

    labels_order = ["adversarial", "cooperative", "neutral"]
    colors = {
        "adversarial": ("#F44336", "#FFCDD2"),
        "cooperative": ("#4CAF50", "#C8E6C9"),
        "neutral":     ("#2196F3", "#BBDEFB"),
    }
    line_styles = {"Baseline": "--", "Transformer": "-"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("ROC Curves — One vs Rest (Baseline vs Transformer)", fontsize=13, fontweight="bold")

    for ax_idx, cls in enumerate(labels_order):
        ax = axes[ax_idx]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (AUC=0.50)")

        for model_name, pred_path in pred_paths.items():
            if not pred_path.exists():
                continue
            gold, p_adv, p_coop, p_neut = [], [], [], []
            with open(pred_path, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    gold.append(row["true_label"].strip().lower())
                    p_adv.append(float(row["p_adversarial"]))
                    p_coop.append(float(row["p_cooperative"]))
                    p_neut.append(float(row["p_neutral"]))

            prob_map = {
                "adversarial": p_adv,
                "cooperative": p_coop,
                "neutral": p_neut,
            }
            binary = [1 if g == cls else 0 for g in gold]
            fpr, tpr, _ = roc_curve(binary, prob_map[cls])
            roc_auc = auc(fpr, tpr)
            color = colors[cls][0] if model_name == "Transformer" else colors[cls][1]
            ax.plot(fpr, tpr,
                    linestyle=line_styles[model_name],
                    color=color,
                    linewidth=2.5,
                    label=f"{model_name} (AUC={roc_auc:.3f})")

        ax.set_title(f"{cls.capitalize()} vs Rest")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)

    plt.tight_layout()
    out = out_dir / "roc_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


def plot_precision_recall_curves(pred_paths: dict, out_dir: Path) -> None:
    """Precision-Recall curves — more informative than ROC for imbalanced classes."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        return

    from sklearn.metrics import precision_recall_curve, average_precision_score

    labels_order = ["adversarial", "cooperative", "neutral"]
    line_styles = {"Baseline": "--", "Transformer": "-"}
    colors = {"Baseline": "#FF9800", "Transformer": "#2196F3"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Precision-Recall Curves — One vs Rest", fontsize=13, fontweight="bold")

    for ax_idx, cls in enumerate(labels_order):
        ax = axes[ax_idx]
        for model_name, pred_path in pred_paths.items():
            if not pred_path.exists():
                continue
            gold, p_adv, p_coop, p_neut = [], [], [], []
            with open(pred_path, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    gold.append(row["true_label"].strip().lower())
                    p_adv.append(float(row["p_adversarial"]))
                    p_coop.append(float(row["p_cooperative"]))
                    p_neut.append(float(row["p_neutral"]))

            prob_map = {"adversarial": p_adv, "cooperative": p_coop, "neutral": p_neut}
            binary = [1 if g == cls else 0 for g in gold]
            prec, rec, _ = precision_recall_curve(binary, prob_map[cls])
            ap = average_precision_score(binary, prob_map[cls])
            ax.plot(rec, prec,
                    linestyle=line_styles[model_name],
                    color=colors[model_name],
                    linewidth=2.5,
                    label=f"{model_name} (AP={ap:.3f})")

        ax.set_title(f"{cls.capitalize()} vs Rest")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)

    plt.tight_layout()
    out = out_dir / "precision_recall_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


if __name__ == "__main__":
    main()
