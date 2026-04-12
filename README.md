# TradePulse — Bilateral economic news (NLP track)

## Primary: supervised NLP pipeline (current work)

**Step 1 — Data collection (live RSS, four target pairs)**

Target pairs are fixed in `nlp/corpus_pairs.py`: **IN–CN, IN–US, CN–US, IN–RU** (edit there if your advisor wants a different slice).

```bash
pip install -r requirements.txt
python scripts/collect_corpus.py              # writes data/raw_headlines.csv
python scripts/collect_corpus.py --feeds 50 --merge   # grow corpus over multiple runs
```

**Live sources:** (1) wire RSS in `core/config.py`, (2) **four Google News search RSS feeds** in `nlp/supplemental_feeds.py` so each headline is tied to one target pair (wire items alone rarely mention both countries). Use `--no-gnews` to disable (2). **Both** paths use the same max age: `MAX_ARTICLE_AGE_DAYS` in `core/config.py` (**90 days ≈ last 3 months**). Override per run with `--days N` if needed.

Output schema: `id`, `headline`, `country_1`, `country_2`, `source`, `url`, `published_at`, `text` (UTF-8 BOM for Excel). If the default path is locked, the script writes `data/raw_headlines_YYYYMMDD_HHMMSS.csv`.

**Step 2 — Labeling (human)**  
Build a sheet with an empty `label` column, then fill each row with exactly: `cooperative` | `neutral` | `adversarial` (wording only; unclear → neutral).

```bash
# From newest raw_headlines*.csv in data/
python scripts/init_labeled_template.py --latest
# Or from a specific file:
python scripts/init_labeled_template.py -i data/raw_headlines_20260412_040641.csv
```

Edit **`data/labeled_dataset.csv`** in Excel/Sheets (save as same UTF-8 CSV if possible).

**Step 3 — Train / test split (after labels are filled)**

```bash
pip install scikit-learn   # if not already from requirements.txt
python scripts/split_data.py              # → data/train.csv, data/test.csv (80/20 stratified)
python scripts/split_data.py --test-size 0.25
```

**Step 4+ (next implementation):** TF-IDF + Logistic Regression baseline, then transformer fine-tune + `scripts/evaluate.py`.

---

## Legacy: RSS + FinBERT + trade snapshot (optional)

| Stage | Implementation |
|--------|------------------|
| Ingest | RSS (`core/config.py`), **90-day** window (~3 months) |
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
