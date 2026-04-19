"""
Attention visualization for DistilBERT.

Extracts attention weights from the last layer and shows which tokens
the model focuses on when making adversarial/cooperative/neutral predictions.

Usage:
    python scripts/attention_viz.py
    python scripts/attention_viz.py --text "India and China sign trade deal" --pair IN-CN
    python scripts/attention_viz.py --examples 5  # visualize 5 test examples

Outputs:
    results/attention/attention_<label>_<pair>.png  — heatmap per example
    results/attention/top_words_summary.png         — top attention words per class
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from nlp.inference import load_transformer
from nlp.label_mapping import LABELS, LABEL2ID

RESULTS = ROOT / "results" / "attention"
DATA = ROOT / "data"

LABEL_COLORS = {
    "adversarial": "#F44336",
    "cooperative": "#4CAF50",
    "neutral":     "#2196F3",
}


def get_attention_and_prediction(text: str) -> dict:
    """
    Run forward pass with output_attentions=True.
    Returns tokens, averaged attention weights, and prediction.
    """
    tokenizer, model = load_transformer()

    enc = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=256, padding=True,
    )
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

    with torch.no_grad():
        outputs = model(**enc, output_attentions=True)

    logits = outputs.logits
    # Temperature scaling
    temp_path = ROOT / "models" / "transformer_bilateral" / "temperature.json"
    if temp_path.exists():
        import json
        T = float(json.loads(temp_path.read_text()).get("temperature", 1.0))
        logits = logits / T

    probs = torch.softmax(logits, dim=-1)[0].numpy()
    pred_idx = int(np.argmax(probs))
    pred_label = LABELS[pred_idx]
    confidence = float(probs[pred_idx])

    # attentions: tuple of (n_layers,) each shape (1, n_heads, seq_len, seq_len)
    # DistilBERT has 6 layers, 12 heads
    # Average across all layers and heads → (seq_len, seq_len)
    attentions = outputs.attentions
    avg_attention = torch.stack(attentions).squeeze(1).mean(dim=(0, 1)).numpy()

    # CLS token attention to all other tokens = what the model "looks at" for classification
    cls_attention = avg_attention[0]  # shape: (seq_len,)

    # Clean tokens (remove ## subword markers, special tokens)
    clean_tokens = []
    clean_attn = []
    for tok, attn in zip(tokens, cls_attention):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        clean_tok = tok.replace("##", "")
        clean_tokens.append(clean_tok)
        clean_attn.append(float(attn))

    # Normalize
    total = sum(clean_attn) or 1.0
    clean_attn = [a / total for a in clean_attn]

    return {
        "text": text,
        "tokens": clean_tokens,
        "attention": clean_attn,
        "pred_label": pred_label,
        "confidence": confidence,
        "probs": {LABELS[i]: float(probs[i]) for i in range(len(LABELS))},
    }


def plot_attention_heatmap(result: dict, out_path: Path, title: str = "") -> None:
    """Plot token attention as a horizontal heatmap."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("  [skip] pip install matplotlib")
        return

    tokens = result["tokens"]
    attn = result["attention"]
    label = result["pred_label"]
    conf = result["confidence"]
    color = LABEL_COLORS[label]

    # Limit to top 30 tokens for readability
    if len(tokens) > 30:
        top_idx = sorted(range(len(attn)), key=lambda i: attn[i], reverse=True)[:30]
        top_idx = sorted(top_idx)
        tokens = [tokens[i] for i in top_idx]
        attn = [attn[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(max(10, len(tokens) * 0.5), 2.5))

    attn_arr = np.array(attn).reshape(1, -1)
    im = ax.imshow(attn_arr, cmap="YlOrRd", aspect="auto", vmin=0)

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    ax.set_yticks([])

    # Color-code by attention intensity
    for i, (tok, a) in enumerate(zip(tokens, attn)):
        text_color = "white" if a > 0.6 * max(attn) else "black"
        ax.text(i, 0, f"{a:.2f}", ha="center", va="center",
                fontsize=7, color=text_color, fontweight="bold")

    headline = f'Prediction: {label.upper()} ({conf*100:.1f}%)'
    ax.set_title(
        f"{title or result['text'][:80]}\n{headline}",
        fontsize=10, color=color, fontweight="bold"
    )
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.3, shrink=0.5, label="Attention weight")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_top_words_summary(examples: list[dict], out_dir: Path) -> None:
    """
    Aggregate top attention words per class across multiple examples.
    Shows which words the model consistently focuses on for each label.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
        from collections import defaultdict
    except ImportError:
        return

    from collections import defaultdict

    # Aggregate attention per token per label
    label_word_attn: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    label_word_count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    stopwords = {"the", "a", "an", "and", "or", "of", "in", "to", "is", "are",
                 "was", "were", "be", "been", "has", "have", "had", "with", "for",
                 "on", "at", "by", "from", "as", "its", "it", "this", "that", "s"}

    for ex in examples:
        label = ex["pred_label"]
        for tok, attn in zip(ex["tokens"], ex["attention"]):
            tok_clean = tok.lower().strip(".,!?;:'\"")
            if len(tok_clean) < 3 or tok_clean in stopwords:
                continue
            label_word_attn[label][tok_clean] += attn
            label_word_count[label][tok_clean] += 1

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Top Attention Words per Class — DistilBERT", fontsize=13, fontweight="bold")

    for ax, label in zip(axes, LABELS):
        word_scores = label_word_attn[label]
        if not word_scores:
            ax.set_title(f"{label.capitalize()}\n(no examples)")
            continue

        # Average attention per word
        avg_scores = {w: word_scores[w] / label_word_count[label][w]
                      for w in word_scores}
        top = sorted(avg_scores.items(), key=lambda x: -x[1])[:15]
        words = [t[0] for t in top]
        scores = [t[1] for t in top]

        color = LABEL_COLORS[label]
        bars = ax.barh(range(len(words)), scores, color=color, alpha=0.8)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=10)
        ax.invert_yaxis()
        ax.set_title(f"{label.capitalize()}", fontsize=12, color=color, fontweight="bold")
        ax.set_xlabel("Avg attention weight")
        ax.grid(True, alpha=0.3, axis="x")

        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{score:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    out = out_dir / "top_words_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Attention visualization for DistilBERT")
    ap.add_argument("--text", default=None, help="Single headline to visualize")
    ap.add_argument("--pair", default="CN-US", help="Bilateral pair e.g. IN-CN")
    ap.add_argument("--examples", type=int, default=9,
                    help="Number of test examples to visualize (3 per class)")
    ap.add_argument("--test-csv", type=Path, default=DATA / "test.csv")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.text:
        # Single headline mode
        c1, c2 = args.pair.upper().split("-") if "-" in args.pair else ("", "")
        input_text = f"{c1}-{c2}: {args.text}" if c1 and c2 else args.text
        print(f"Analyzing: {input_text}")
        result = get_attention_and_prediction(input_text)
        print(f"Prediction: {result['pred_label']} ({result['confidence']*100:.1f}%)")
        out = RESULTS / f"attention_single.png"
        plot_attention_heatmap(result, out, title=args.text[:60])
        print(f"Saved → {out}")
        return

    # Multi-example mode: pick 3 examples per class from test set
    print(f"Loading test examples from {args.test_csv.name}...")
    with open(args.test_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # Pick 3 per class
    per_class = args.examples // 3
    selected = []
    for label in LABELS:
        class_rows = [r for r in rows if r.get("label", "").strip().lower() == label]
        selected.extend(class_rows[:per_class])

    print(f"Visualizing {len(selected)} examples ({per_class} per class)...")
    all_results = []

    for i, row in enumerate(selected):
        c1 = (row.get("country_1") or "").strip().upper()
        c2 = (row.get("country_2") or "").strip().upper()
        text = (row.get("text") or row.get("headline") or "").strip()
        true_label = row.get("label", "").strip().lower()
        input_text = f"{c1}-{c2}: {text}" if c1 and c2 else text

        print(f"  [{i+1}/{len(selected)}] {c1}-{c2} | true={true_label} | {text[:50]}...")
        result = get_attention_and_prediction(input_text)
        result["true_label"] = true_label
        result["pair"] = f"{c1}-{c2}"
        all_results.append(result)

        match = "✓" if result["pred_label"] == true_label else "✗"
        print(f"    {match} pred={result['pred_label']} ({result['confidence']*100:.1f}%)")

        out = RESULTS / f"attention_{true_label}_{c1}{c2}_{i}.png"
        plot_attention_heatmap(result, out, title=f"[{c1}-{c2}] {text[:60]}")

    print(f"\nGenerating top words summary...")
    plot_top_words_summary(all_results, RESULTS)

    print(f"\nAll attention plots saved to {RESULTS}/")
    print("Key file: results/attention/top_words_summary.png")


if __name__ == "__main__":
    main()
