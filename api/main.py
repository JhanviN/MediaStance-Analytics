"""
FastAPI: POST /classify

Run from repo root:
  pip install fastapi uvicorn apscheduler
  uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Literal, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.routes_analytics import router as analytics_router
from nlp.inference import combine_headline_body, predict_baseline, predict_transformer, get_attention_weights
from nlp.pair_utils import parse_pair, pair_key_from_codes
from nlp.predictions_sqlite import connect, init_predictions_db, insert_prediction

logger = logging.getLogger("mediastance")

# ── Background scheduler ──────────────────────────────────────────────────────
def _run_live_pipeline() -> None:
    """Called by scheduler every N minutes — collect fresh headlines and predict."""
    try:
        sys.path.insert(0, str(ROOT / "pipeline"))
        from live_pipeline import run_once
        db_conn = connect(ROOT / "data" / "predictions.db")
        init_predictions_db(db_conn)
        added = run_once(db_conn, model="baseline")
        db_conn.close()
        logger.info(f"Scheduled pipeline run: {added} new predictions added")
    except Exception as e:
        logger.error(f"Scheduled pipeline error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Download DB from HF Dataset, start background scheduler, stop on shutdown."""
    logger.info(f"Starting up. ROOT={ROOT}")

    # ── DB sync: download latest predictions.db on startup ───────────────────
    try:
        # Ensure repo root on path (belt-and-suspenders for Docker)
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from core.db_sync import init_sync, start_background_sync  # noqa: E402
        init_sync()
        start_background_sync()
        logger.info("DB sync initialized successfully")
    except ModuleNotFoundError as e:
        # Log the full sys.path so we can debug exactly what's missing
        logger.warning(f"DB sync failed — {e}. sys.path={sys.path[:5]}")
    except Exception as e:
        logger.warning(f"DB sync init failed: {e} (type: {type(e).__name__})")

    # ── Download transformer model from HF Hub if missing ────────────────────
    _model_dir = ROOT / "models" / "transformer_bilateral"
    if not (_model_dir / "model.safetensors").exists():
        hf_repo = os.environ.get("HF_MODEL_REPO", "JhanviN/mediastance-deploy").strip()
        if hf_repo:
            try:
                from huggingface_hub import snapshot_download
                logger.info(f"Downloading transformer model from {hf_repo}...")
                snapshot_download(
                    repo_id=hf_repo,
                    local_dir=str(_model_dir),
                    token=os.environ.get("HF_TOKEN", "").strip() or None,
                )
                logger.info("Transformer model downloaded.")
            except Exception as e:
                logger.warning(f"Could not download transformer model: {e}")

    # ── Background pipeline scheduler ────────────────────────────────────────
    interval = int(os.environ.get("PIPELINE_INTERVAL_MINUTES", "60"))
    scheduler = None

    if interval > 0:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            scheduler = BackgroundScheduler()
            scheduler.add_job(
                _run_live_pipeline,
                "interval",
                minutes=interval,
                id="live_pipeline",
                replace_existing=True,
            )
            scheduler.start()
            logger.info(f"Live pipeline scheduler started — runs every {interval} min")
        except ImportError:
            logger.warning("apscheduler not installed — scheduler disabled.")

    yield  # app runs here

    if scheduler and scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")


app = FastAPI(
    title="MediaStance Analytics API",
    version="1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(analytics_router)


class ClassifyRequest(BaseModel):
    text: str = Field(..., description="Headline or full text (same as training)")
    body: Optional[str] = Field(None, description="Optional extra snippet")
    headline: Optional[str] = Field(None, description="Short label for DB when saving")
    pair: Optional[str] = Field(
        None,
        description="Bilateral pair e.g. IN-US or US-CN (sorted for DB); overrides country_1/2 when set",
    )
    country_1: Optional[str] = Field(None, description="ISO2 e.g. IN (ignored if pair is set)")
    country_2: Optional[str] = Field(None, description="ISO2 e.g. US")
    model: Literal["baseline", "transformer", "both"] = "both"
    save: bool = Field(False, description="If true, append rows to data/predictions.db")


class ProbBlock(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]


class ClassifyResponse(BaseModel):
    text_used: str
    pair: Optional[str] = Field(None, description="Canonical pair key if countries were resolved")
    baseline: Optional[ProbBlock] = None
    transformer: Optional[ProbBlock] = None


class BatchClassifyItem(BaseModel):
    text: str
    pair: Optional[str] = None
    save: bool = False


class BatchClassifyRequest(BaseModel):
    items: list[BatchClassifyItem]
    model: Literal["baseline", "transformer", "both"] = "both"


class AttentionRequest(BaseModel):
    text: str = Field(..., description="Headline or text to visualize")
    pair: Optional[str] = Field(None, description="Bilateral pair e.g. IN-US")


class AttentionResponse(BaseModel):
    tokens: list[str]
    weights: list[float]
    label: str
    confidence: float


@app.post("/attention", response_model=AttentionResponse)
def attention(req: AttentionRequest):
    t = req.text.strip()
    if not t:
        raise HTTPException(400, "text is required")
    try:
        tokens, weights, label, confidence = get_attention_weights(t)
        return AttentionResponse(tokens=tokens, weights=weights, label=label, confidence=confidence)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/classify", include_in_schema=False)
def classify_get_hint():
    """Opening /classify in a browser uses GET; classification requires POST."""
    return {
        "note": "/classify expects POST with a JSON body (a browser address bar sends GET).",
        "try": "POST from Postman/curl, or use GET /docs and run POST /classify there.",
        "example_body": {
            "text": "India and the United States hold trade talks",
            "pair": "IN-US",
            "model": "baseline",
            "save": False,
        },
    }


def _infer_classify(text: str, body: Optional[str], model: str) -> tuple[str, ClassifyResponse]:
    t = text.strip()
    if not t:
        raise HTTPException(400, "text is required")
    if body and body.strip():
        t = combine_headline_body(t, body.strip())
    resp = ClassifyResponse(text_used=t)
    try:
        if model in ("baseline", "both"):
            lab, conf, probs = predict_baseline(t)
            resp.baseline = ProbBlock(label=lab, confidence=round(conf, 4), probabilities=probs)
        if model in ("transformer", "both"):
            lab, conf, probs = predict_transformer(t)
            resp.transformer = ProbBlock(label=lab, confidence=round(conf, 4), probabilities=probs)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    return t, resp


def _resolve_countries(req: ClassifyRequest) -> tuple[Optional[str], Optional[str]]:
    if req.pair and str(req.pair).strip():
        try:
            a, b = parse_pair(req.pair)
            return a, b
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
    c1 = req.country_1.strip().upper() if req.country_1 else None
    c2 = req.country_2.strip().upper() if req.country_2 else None
    return c1, c2


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    t, resp = _infer_classify(req.text, req.body, req.model)
    headline = (req.headline or t[:200]).strip()
    c1, c2 = _resolve_countries(req)
    if c1 and c2:
        resp.pair = pair_key_from_codes(c1, c2)

    if req.save:
        conn = connect()
        init_predictions_db(conn)
        if resp.baseline:
            insert_prediction(
                conn,
                headline=headline,
                text_used=t,
                country_1=c1,
                country_2=c2,
                model="baseline",
                label=resp.baseline.label,
                confidence=resp.baseline.confidence,
                probs=resp.baseline.probabilities,
            )
        if resp.transformer:
            insert_prediction(
                conn,
                headline=headline,
                text_used=t,
                country_1=c1,
                country_2=c2,
                model="transformer",
                label=resp.transformer.label,
                confidence=resp.transformer.confidence,
                probs=resp.transformer.probabilities,
            )
        conn.close()

    return resp


@app.post("/classify/batch")
def classify_batch(req: BatchClassifyRequest):
    """Up to 50 headlines; same model for all items."""
    if len(req.items) > 50:
        raise HTTPException(400, "at most 50 items per batch")
    out: list[dict] = []
    for it in req.items:
        sub = ClassifyRequest(
            text=it.text,
            pair=it.pair,
            model=req.model,
            save=it.save,
        )
        try:
            r = classify(sub)
            out.append(r.model_dump())
        except HTTPException as e:
            out.append({"text": it.text, "error": e.detail, "status_code": e.status_code})
    return {"results": out, "count": len(out)}
