import csv
from collections import Counter
from pathlib import Path

p = Path("data/synthetic_raw.csv")
if not p.exists():
    print("data/synthetic_raw.csv does not exist — nothing was saved.")
else:
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    print(f"Total rows saved: {len(rows)}")
    pairs = Counter(f"{r['country_1']}-{r['country_2']}" for r in rows)
    labels = Counter(r['label'] for r in rows)
    print(f"Label dist: {dict(labels)}")
    print("Pairs completed:")
    for k, v in sorted(pairs.items()):
        print(f"  {k}: {v}")
    # Show which pairs are missing (need to be re-run)
    from nlp.corpus_pairs import CORPUS_TARGET_PAIRS
    all_pairs = set(f"{min(a,b)}-{max(a,b)}" for a,b in CORPUS_TARGET_PAIRS)
    done = set(pairs.keys())
    missing = all_pairs - done
    print(f"\nPairs still needed: {missing if missing else 'ALL DONE'}")
