# TradePulse — Bilateral economic news (NLP track)

## Primary: supervised NLP pipeline (current work)

**Step 1 — Data collection (live RSS, four target pairs)**

Target pairs are fixed in `nlp/corpus_pairs.py`: **IN–CN, IN–US, CN–US, IN–RU** (edit there if your advisor wants a different slice).

```bash
pip install -r requirements.txt
python scripts/collect_corpus.py              # writes data/raw_headlines.csv
python scripts/collect_corpus.py --feeds 50 --merge   # grow corpus over multiple runs
```

**Live sources:** (1) wire RSS in `core/config.py`, (2) **four Google News search RSS feeds** in `nlp/supplemental_feeds.py` so each headline is tied to one target pair (wire items alone rarely mention both countries). Use `--no-gnews` to disable (2). Use `--days 90` if the wire slice is too thin.

Output schema: `id`, `headline`, `country_1`, `country_2`, `source`, `url`, `published_at`, `text` (UTF-8 BOM CSV for Excel).

**Next steps (not automated yet):** add column `label` → `data/labeled_dataset.csv`, then `train.csv` / `test.csv`, TF-IDF baseline, transformer fine-tune, evaluation scripts (see conversation spec).

---

## Legacy: RSS + FinBERT + trade snapshot (optional)

| Stage | Implementation |
|--------|------------------|
| Ingest | RSS (`core/config.py`), 30-day window |
| NLP | FinBERT zero-shot (`core/sentiment.py`) |
| Structure | `data/trade_snapshot.json` |
| Score | Two-pillar 0–10 in `core/leverage_engine.py` (not part of the NLP-classification thesis) |

```bash
python run_pipeline.py --no-wb
streamlit run demo/demo_app.py
```

## Layout (excerpt)

```
nlp/corpus_pairs.py       # four target bilateral pairs
scripts/collect_corpus.py # Step 1 → data/raw_headlines.csv
data/raw_headlines.csv    # created by collector (UTF-8 BOM for Excel)
core/news_fetcher.py      # RSS fetch + country mention heuristics
```

## Expanding beyond four pairs

Add a tuple to `CORPUS_TARGET_PAIRS` **and** ensure `data/trade_snapshot.json` has a block if you still run the legacy leverage pipeline; the NLP track only needs the collector pair list.

## Legacy trade snapshot

Edit `data/trade_snapshot.json` only with cited statistics; used by `run_pipeline.py` / Streamlit path only.
