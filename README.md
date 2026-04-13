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
Fill **`data/labeled_dataset.csv`** column `label` with exactly: `cooperative` | `neutral` | `adversarial` (wording only; unclear → neutral).

**First time** (create sheet from raw):

```bash
python scripts/sync_labeled_dataset.py --latest
# or: python scripts/init_labeled_template.py --latest
```

**After you fetch more raw data** (`collect_corpus.py --merge` …): **sync again** — same ids **keep your labels**; new ids get an empty `label`; ids that disappeared from this raw pull but were already labeled are **kept at the bottom** so you never lose work.

```bash
python scripts/collect_corpus.py --feeds 80 --merge
python scripts/sync_labeled_dataset.py --latest
```

Close `labeled_dataset.csv` in Excel before sync/split if you see permission errors.

**Cleaner `text` + optional full article body**  
RSS/Google only ship short snippets — not the full article. The collector now collapses **near-duplicate** title/summary using word overlap (e.g. `… - ThePrint` vs `… ThePrint`). To pull **main body text** from each URL (slow, ~1s/row, some sites block bots):

```bash
pip install trafilatura
python scripts/enrich_corpus_bodies.py -i data/raw_headlines.csv -o data/raw_headlines_enriched.csv --limit 10
```

Use `-o` as the input to `sync_labeled_dataset` / labeling when satisfied; `--limit` is for testing.

**Step 3 — Train / test split (after labels are filled)**

```bash
python scripts/split_data.py              # → data/train.csv, data/test.csv (80/20 stratified)
python scripts/split_data.py --test-size 0.25
```

**Step 4 — Train models**

```bash
python scripts/train_baseline.py          # TF-IDF + Logistic Regression → models/baseline_tfidf_lr.joblib
python scripts/train_transformer.py       # Fine-tune DistilBERT (3 epochs default; use --epochs 1 for a quick test)
python scripts/train_transformer.py --epochs 3 --batch-size 8
```

**Step 5 — Outputs:** `results/baseline_test_predictions.csv`, `results/transformer_test_predictions.csv`

**Step 6 — Evaluation table**

```bash
python scripts/evaluate.py                # → results/evaluation_report.md
```

**Step 7 — Error analysis**

```bash
python scripts/error_analysis.py
python scripts/error_analysis.py --pred results/baseline_test_predictions.csv
```

**Step 8 — Live prediction (CLI), SQLite log, weekly report, API**

```bash
# One headline → JSON with label + per-class probabilities (baseline and/or transformer)
python scripts/predict.py -t "India and US hold trade talks amid tariff dispute"
python scripts/predict.py -t "Headline here" --body "Optional extra snippet" --model baseline

# Persist each run to SQLite (default path: data/predictions.db)
python scripts/predict.py -t "..." --country1 IN --country2 US --save-db

# Rolling summary: last N days vs prior N days (UTC), from predictions.db
python scripts/generate_weekly_report.py
python scripts/generate_weekly_report.py --days 7 --top 15

# HTTP API (same models as the CLI)
pip install fastapi uvicorn
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
# POST /classify  — body e.g. {"text":"...", "pair":"IN-US", "model":"baseline", "save": false}
# Analytics (reads data/predictions.db; aggregates model=baseline by default):
#   GET /summary?pair=IN-US   GET /summary/all   GET /distribution?pair=CN-US
#   GET /trends?pair=IN-US&rolling=7   GET /headlines?pair=IN-US&label=adversarial
#   GET /alerts   GET /compare?pair1=IN-US&pair2=CN-US
# POST /classify/batch — up to 50 items: {"items":[{"text":"...","pair":"IN-US"}], "model":"baseline"}
# OpenAPI: http://127.0.0.1:8000/docs
# Full route list + headers + example responses: docs/API_ROUTES.md
```

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








When you add more raw rows: `collect_corpus.py --merge`, then `sync_labeled_dataset.py --latest`, label only new empty rows, then `split_data.py` and retrain/eval as needed. Close the CSV in Excel on Windows if you see file lock errors.