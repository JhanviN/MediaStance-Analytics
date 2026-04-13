"""Fixed label order for sklearn + Hugging Face (id stable across scripts)."""

from __future__ import annotations

# Sorted for stable IDs: 0, 1, 2
LABELS: list[str] = ["adversarial", "cooperative", "neutral"]
LABEL2ID: dict[str, int] = {lab: i for i, lab in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: lab for i, lab in enumerate(LABELS)}
