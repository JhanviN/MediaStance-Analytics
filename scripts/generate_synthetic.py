"""
Synthetic bilateral headline generator.

Supports:
  - LM Studio (local, free) — recommended
  - OpenAI API
  - Google Gemini API

Usage:
    # LM Studio (load a model in LM Studio first, then):
    python scripts/generate_synthetic.py --backend lmstudio --per-class 200 --out data/synthetic_raw.csv

    # OpenAI:
    set OPENAI_API_KEY=sk-...
    python scripts/generate_synthetic.py --backend openai --per-class 200 --out data/synthetic_raw.csv

    # Gemini:
    set GEMINI_API_KEY=...
    python scripts/generate_synthetic.py --backend gemini --per-class 200 --out data/synthetic_raw.csv

Output schema matches raw_headlines.csv / labeled_dataset.csv.
"""

from __future__ import annotations

import dotenv; dotenv.load_dotenv()

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.corpus_pairs import CORPUS_TARGET_PAIRS  # noqa: E402

FIELDNAMES = [
    "id", "headline", "country_1", "country_2",
    "source", "url", "published_at", "text", "label",
]

# ── Country full names for prompts ────────────────────────────────────────────
COUNTRY_NAMES = {
    "IN": "India", "CN": "China", "US": "United States",
    "RU": "Russia", "PK": "Pakistan", "IR": "Iran", "IL": "Israel",
}

# ── News sources to randomly assign (realistic variety) ──────────────────────
SOURCES = [
    "Reuters", "Bloomberg", "The Economic Times", "South China Morning Post",
    "Al Jazeera", "BBC Business", "Financial Times", "NDTV", "The Hindu",
    "Nikkei Asia", "The Times of India", "Business Standard", "CNBC",
    "The Guardian", "AP News", "Mint", "LiveMint", "The Wire",
]

# ── Per-class prompt templates ────────────────────────────────────────────────
LABEL_PROMPTS = {
    "adversarial": """Generate {n} realistic news headlines about the economic/trade relationship between {country_a} and {country_b} that signal ADVERSARIAL or HOSTILE dynamics.

Examples of adversarial signals: tariffs, sanctions, trade war, ban, expulsion, retaliation, coercion, dispute, tension, threat, probe, restriction, dumping accusation, espionage concern, decoupling.

Rules:
- Each headline must be a single line, realistic news headline style
- Mix short (5-8 words) and longer (10-15 words) headlines
- Include source names at the end like "- Reuters" or "- Bloomberg" for some (not all)
- Some headlines should be ambiguous but lean adversarial
- Vary the framing: some about tariffs, some about diplomatic friction, some about trade imbalance, some about security concerns
- Do NOT number the headlines
- Output ONLY the headlines, one per line, nothing else""",

    "cooperative": """Generate {n} realistic news headlines about the economic/trade relationship between {country_a} and {country_b} that signal COOPERATIVE or POSITIVE dynamics.

Examples of cooperative signals: trade deal, partnership, agreement, investment, joint venture, cooperation, summit, bilateral talks progress, MOU signed, market access, diplomatic reset, trade growth.

Rules:
- Each headline must be a single line, realistic news headline style
- Mix short (5-8 words) and longer (10-15 words) headlines
- Include source names at the end like "- Reuters" or "- Bloomberg" for some (not all)
- Some headlines should be cautiously optimistic but lean cooperative
- Vary the framing: some about deals, some about investment flows, some about diplomatic progress, some about trade milestones
- Do NOT number the headlines
- Output ONLY the headlines, one per line, nothing else""",

    "neutral": """Generate {n} realistic news headlines about the economic/trade relationship between {country_a} and {country_b} that are NEUTRAL or AMBIGUOUS in stance.

Examples of neutral signals: trade data release, economic analysis, policy review, expert commentary, market update, trade statistics, bilateral meeting scheduled, ongoing negotiations, mixed signals.

Rules:
- Each headline must be a single line, realistic news headline style
- Mix short (5-8 words) and longer (10-15 words) headlines
- Include source names at the end like "- Reuters" or "- Bloomberg" for some (not all)
- Headlines should be genuinely ambiguous — could go either way
- Vary the framing: some factual data reports, some analysis pieces, some "what to watch" style
- Do NOT number the headlines
- Output ONLY the headlines, one per line, nothing else""",
}

# ── Pair-specific context injections ─────────────────────────────────────────
PAIR_CONTEXT = {
    ("CN", "IN"): "Focus on: trade deficit, border tensions, FDI restrictions, supply chain, Himalayan border, BRI, RCEP, manufacturing competition.",
    ("IN", "US"): "Focus on: trade deal negotiations, tariffs, H1B visas, tech sector, defense cooperation, pharmaceutical exports, market access.",
    ("CN", "US"): "Focus on: tariff war, tech decoupling, semiconductor restrictions, Taiwan, fentanyl, TikTok, supply chain, trade deficit.",
    ("IN", "RU"): "Focus on: Russian oil imports, sanctions pressure, defense deals, S-400, rupee-ruble trade, energy security, Western pressure.",
    ("IN", "PK"): "Focus on: border skirmishes, trade suspension, terrorism accusations, water disputes, Kashmir, ceasefire violations, diplomatic expulsions.",
    ("IN", "IR"): "Focus on: Chabahar port, oil imports, US sanctions on Iran, rupee trade, connectivity corridor, strategic partnership.",
    ("IL", "IN"): "Focus on: defense exports, drone technology, agricultural cooperation, Gaza conflict impact on relations, arms deals.",
    ("CN", "RU"): "Focus on: energy trade, sanctions bypass, yuan-ruble settlement, military cooperation, Ukraine war, strategic partnership.",
    ("CN", "IR"): "Focus on: 25-year cooperation deal, oil imports, BRI, sanctions evasion, nuclear program, strategic alignment.",
    ("CN", "PK"): "Focus on: CPEC, Belt and Road, infrastructure investment, debt trap concerns, military cooperation, Gwadar port.",
    ("RU", "US"): "Focus on: Ukraine sanctions, energy embargo, SWIFT exclusion, diplomatic expulsions, nuclear threats, asset seizures.",
    ("IR", "US"): "Focus on: nuclear deal (JCPOA), oil sanctions, drone attacks, proxy conflicts, diplomatic freeze, prisoner swaps.",
    ("IL", "US"): "Focus on: military aid, Gaza ceasefire pressure, two-state solution, arms transfers, UN vetoes, diplomatic tensions.",
    ("IL", "IR"): "Focus on: direct missile strikes, nuclear program, proxy war, Hezbollah, shadow war, assassination campaigns.",
    ("IR", "RU"): "Focus on: drone supply to Russia, energy cooperation, sanctions alignment, military partnership, anti-Western axis.",
}


def _row_id(headline: str, c1: str, c2: str, label: str) -> str:
    h = hashlib.sha256(f"synth|{headline}|{c1}|{c2}|{label}".encode()).hexdigest()
    return h[:20]


def _call_api(prompt: str, backend: str, model: str, max_retries: int = 3) -> Optional[str]:
    """Unified API caller. Supports lmstudio, openai, gemini."""

    system_msg = (
        "You are a news headline generator for an NLP research dataset. "
        "Generate realistic, varied news headlines exactly as instructed. "
        "Output only the headlines, one per line, nothing else. "
        "No numbering, no bullet points, no explanations."
    )

    # ── LM Studio ─────────────────────────────────────────────────────────────
    if backend == "lmstudio":
        try:
            from openai import OpenAI
        except ImportError:
            print("[Error] Run: pip install openai")
            sys.exit(1)
        lm_url = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1").strip()
        client = OpenAI(base_url=lm_url, api_key="lm-studio")
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        # No system role — merge instruction into user message
                        # (some local models only support user/assistant roles)
                        {
                            "role": "user",
                            "content": (
                                "You are a news headline generator for an NLP research dataset. "
                                "Generate realistic, varied news headlines exactly as instructed. "
                                "Output only the headlines, one per line, nothing else. "
                                "No numbering, no bullet points, no explanations.\n\n"
                                + prompt
                            ),
                        }
                    ],
                    temperature=0.85,
                    max_tokens=2000,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"  [LM Studio attempt {attempt+1}/{max_retries}] {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        return None

    # ── OpenAI ────────────────────────────────────────────────────────────────
    elif backend == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            print("[Error] Run: pip install openai")
            sys.exit(1)
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            print("[Error] OPENAI_API_KEY not set.")
            sys.exit(1)
        client = OpenAI(api_key=api_key)
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.85,
                    max_tokens=2000,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"  [OpenAI attempt {attempt+1}/{max_retries}] {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return None

    # ── Gemini ────────────────────────────────────────────────────────────────
    elif backend == "gemini":
        try:
            import google.generativeai as genai
        except ImportError:
            print("[Error] Run: pip install google-generativeai")
            sys.exit(1)
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            print("[Error] GEMINI_API_KEY not set.")
            sys.exit(1)
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel(model_name=model, system_instruction=system_msg)
        for attempt in range(max_retries):
            try:
                resp = client.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(temperature=0.85, max_output_tokens=2000),
                )
                return resp.text.strip()
            except Exception as e:
                print(f"  [Gemini attempt {attempt+1}/{max_retries}] {e}")
                if attempt < max_retries - 1:
                    wait = 30 if "429" in str(e) or "quota" in str(e).lower() else 2 ** attempt
                    time.sleep(wait)
        return None

    else:
        print(f"[Error] Unknown backend: {backend}. Use lmstudio, openai, or gemini.")
        sys.exit(1)


def generate_for_pair_label(
    pair: Tuple[str, str],
    label: str,
    n: int,
    backend: str,
    model: str,
    batch_size: int = 30,
) -> List[Dict]:
    """Generate `n` synthetic headlines for a (pair, label) combination."""
    c1, c2 = pair
    country_a = COUNTRY_NAMES.get(c1, c1)
    country_b = COUNTRY_NAMES.get(c2, c2)
    context = PAIR_CONTEXT.get(pair, "")

    rows: List[Dict] = []
    remaining = n

    while remaining > 0:
        batch_n = min(batch_size, remaining)
        base_prompt = LABEL_PROMPTS[label].format(
            n=batch_n, country_a=country_a, country_b=country_b
        )
        if context:
            base_prompt += f"\n\nPair-specific context to draw from: {context}"

        print(f"  Generating {batch_n} {label} headlines for {c1}-{c2}...", end=" ", flush=True)
        content = _call_api(base_prompt, backend=backend, model=model)

        if not content:
            print("FAILED (skipping batch)")
            remaining -= batch_n
            continue

        lines = [l.strip() for l in content.splitlines() if l.strip()]

        # Debug: show raw output on first batch to catch format issues
        if not rows and lines:
            print(f"\n  [debug] raw output sample: {lines[:3]}")

        headlines = []
        for l in lines:
            # Strip common list prefixes local models add: "1.", "1)", "-", "*", "•"
            import re as _re
            l = _re.sub(r"^[\d]+[\.\)]\s*", "", l)   # "1. " or "1) "
            l = _re.sub(r"^[-•*]\s*", "", l)          # "- " or "• " or "* "
            l = l.strip()
            if len(l) > 10 and len(l) < 300:
                headlines.append(l)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        import random
        for hl in headlines[:batch_n]:
            rows.append({
                "id": _row_id(hl, c1, c2, label),
                "headline": hl[:500],
                "country_1": c1,
                "country_2": c2,
                "source": f"synthetic-{random.choice(SOURCES)}",
                "url": f"urn:synthetic:{_row_id(hl, c1, c2, label)}",
                "published_at": now,
                "text": hl[:4000],
                "label": label,
            })

        print(f"got {len(headlines[:batch_n])} headlines")
        remaining -= batch_n
        # Small delay — LM Studio needs a moment between requests
        time.sleep(0.5 if backend == "lmstudio" else 1.0)

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic bilateral headlines")
    ap.add_argument(
        "--backend", default="lmstudio",
        choices=["lmstudio", "openai", "gemini"],
        help="API backend to use (default: lmstudio)"
    )
    ap.add_argument(
        "--model", default="mistral-7b-instruct-v0.3",
        help=(
            "Model name. LM Studio: use whatever is loaded (e.g. mistral-7b-instruct-v0.3). "
            "OpenAI: gpt-4o-mini. Gemini: gemini-2.5-flash."
        )
    )
    ap.add_argument(
        "--per-class", type=int, default=200,
        help="Headlines per (pair, label) combination (default 200)"
    )
    ap.add_argument(
        "--pairs", nargs="+", default=None,
        help="Specific pairs e.g. IN-US CN-US (default: all 15 pairs)"
    )
    ap.add_argument(
        "--labels", nargs="+", default=["adversarial", "cooperative", "neutral"],
        choices=["adversarial", "cooperative", "neutral"],
    )
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "synthetic_raw.csv")
    ap.add_argument("--merge", action="store_true", help="Append to existing file")
    ap.add_argument("--batch-size", type=int, default=30, help="Headlines per API call (default 30)")
    args = ap.parse_args()

    # Resolve pairs
    if args.pairs:
        target_pairs = []
        for p in args.pairs:
            parts = p.upper().split("-")
            if len(parts) != 2:
                print(f"[Warning] Invalid pair: {p}, skipping")
                continue
            target_pairs.append(tuple(sorted(parts)))
    else:
        target_pairs = [tuple(sorted([a, b])) for a, b in CORPUS_TARGET_PAIRS]

    total_expected = len(target_pairs) * len(args.labels) * args.per_class
    print(f"Backend:  {args.backend}")
    print(f"Model:    {args.model}")
    print(f"Pairs:    {len(target_pairs)} pairs")
    print(f"Labels:   {args.labels}")
    print(f"Per class: {args.per_class}")
    print(f"Expected: ~{total_expected} headlines\n")

    if args.backend == "lmstudio":
        lm_url = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
        print(f"LM Studio URL: {lm_url}")
        print("Make sure LM Studio is running with a model loaded.\n")

    all_rows: List[Dict] = []
    seen_ids: set = set()

    if args.merge and args.out.exists():
        with open(args.out, newline="", encoding="utf-8-sig") as f:
            existing = list(csv.DictReader(f))
        for r in existing:
            if r.get("id"):
                seen_ids.add(r["id"])
        all_rows.extend(existing)
        print(f"Loaded {len(existing)} existing rows from {args.out.name}\n")

    for pair in target_pairs:
        for label in args.labels:
            print(f"\n[{pair[0]}-{pair[1]}] {label.upper()}")
            new_rows = generate_for_pair_label(
                pair=pair,
                label=label,
                n=args.per_class,
                backend=args.backend,
                model=args.model,
                batch_size=args.batch_size,
            )
            added = 0
            for row in new_rows:
                if row["id"] not in seen_ids:
                    seen_ids.add(row["id"])
                    all_rows.append(row)
                    added += 1
            print(f"  → {added} unique rows added")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row.get(k, "") for k in FIELDNAMES})

    from collections import Counter
    labels_dist = Counter(r.get("label", "") for r in all_rows if r.get("label"))
    pairs_dist = Counter(f"{r.get('country_1','')}-{r.get('country_2','')}" for r in all_rows)
    print(f"\n{'='*50}")
    print(f"Wrote {len(all_rows)} rows → {args.out}")
    print(f"Label dist: {dict(labels_dist)}")
    print(f"Pair dist:  {dict(pairs_dist)}")
    print(f"\nNext: python scripts/merge_augmented_data.py --synthetic {args.out}")


if __name__ == "__main__":
    main()
