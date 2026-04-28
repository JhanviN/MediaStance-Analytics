"""
HF Dataset-backed persistence for predictions.db.

The container filesystem is ephemeral on both HF Spaces and Render free tier.
This module treats a private HF Dataset repo as the canonical store:

  startup  -> download latest DB from HF Dataset repo (if available)
  periodic -> upload current DB back every SYNC_INTERVAL_MINUTES minutes
  shutdown -> best-effort final upload via atexit

Required environment variables:
  HF_TOKEN          -- HF token with write access to the dataset repo
  HF_DATASET_REPO   -- e.g. "JhanviN/mediastance-db"

Optional:
  SYNC_INTERVAL_MINUTES  -- how often to push DB back (default: 30)
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import threading
from pathlib import Path

logger = logging.getLogger("mediastance.db_sync")

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "predictions.db"
DB_FILENAME = "predictions.db"

_sync_thread: threading.Thread | None = None
_stop_event = threading.Event()


def _get_config() -> tuple[str | None, str | None, int]:
    token = os.environ.get("HF_TOKEN", "").strip() or None
    dataset_repo = os.environ.get("HF_DATASET_REPO", "").strip() or None
    interval = int(os.environ.get("SYNC_INTERVAL_MINUTES", "30"))
    return token, dataset_repo, interval


def download_db() -> bool:
    """
    Download predictions.db from HF Dataset repo.
    Returns True if successful.
    """
    token, dataset_repo, _ = _get_config()
    if not token or not dataset_repo:
        logger.info("db_sync: HF_TOKEN or HF_DATASET_REPO not set -- skipping download")
        return False

    try:
        from huggingface_hub import hf_hub_download
        logger.info(f"db_sync: downloading {DB_FILENAME} from {dataset_repo} ...")

        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            filename=DB_FILENAME,
            local_dir=str(DB_PATH.parent),
            local_dir_use_symlinks=False,
            token=token,
        )

        if downloaded and Path(downloaded).exists() and Path(downloaded).stat().st_size > 0:
            if str(Path(downloaded).resolve()) != str(DB_PATH.resolve()):
                shutil.move(downloaded, str(DB_PATH))
            size_mb = DB_PATH.stat().st_size / 1024 / 1024
            logger.info(f"db_sync: downloaded {size_mb:.1f} MB -> {DB_PATH}")
            return True
        else:
            logger.warning("db_sync: downloaded file is empty or missing")
            return False

    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower() or "EntryNotFoundError" in type(e).__name__:
            logger.info("db_sync: no DB in dataset repo yet -- will upload after first run")
        else:
            logger.warning(f"db_sync: download failed: {e}")
        return False


def upload_db() -> bool:
    """Upload current predictions.db to HF Dataset repo."""
    token, dataset_repo, _ = _get_config()
    if not token or not dataset_repo:
        return False

    if not DB_PATH.exists():
        logger.warning("db_sync: predictions.db not found, skipping upload")
        return False

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        size_mb = DB_PATH.stat().st_size / 1024 / 1024
        logger.info(f"db_sync: uploading {size_mb:.1f} MB -> {dataset_repo} ...")
        api.upload_file(
            path_or_fileobj=str(DB_PATH),
            path_in_repo=DB_FILENAME,
            repo_id=dataset_repo,
            repo_type="dataset",
            commit_message="sync: predictions.db update",
        )
        logger.info("db_sync: upload complete")
        return True
    except Exception as e:
        logger.warning(f"db_sync: upload failed: {e}")
        return False


def _sync_loop(interval_minutes: int) -> None:
    """Background thread: upload DB every interval_minutes."""
    logger.info(f"db_sync: background sync started -- every {interval_minutes} min")
    while not _stop_event.wait(timeout=interval_minutes * 60):
        upload_db()
    logger.info("db_sync: background sync stopped")


def start_background_sync() -> None:
    """Start the periodic upload thread. No-op if env vars not set."""
    global _sync_thread

    token, dataset_repo, interval = _get_config()
    if not token or not dataset_repo:
        return

    if _sync_thread and _sync_thread.is_alive():
        return

    _stop_event.clear()
    _sync_thread = threading.Thread(
        target=_sync_loop,
        args=(interval,),
        daemon=True,
        name="db-sync",
    )
    _sync_thread.start()
    atexit.register(upload_db)
    logger.info(f"db_sync: sync thread started, interval={interval}min, repo={dataset_repo}")


def stop_background_sync() -> None:
    """Signal the sync thread to stop."""
    _stop_event.set()


def init_sync() -> bool:
    """
    Download DB on startup and start background upload thread.
    Runs on any platform (HF Spaces, Render, local) when env vars are set.
    Returns True if a fresh DB was downloaded.
    """
    token, dataset_repo, _ = _get_config()
    if not token or not dataset_repo:
        logger.info("db_sync: HF_TOKEN or HF_DATASET_REPO not set -- sync disabled")
        return False

    downloaded = download_db()
    start_background_sync()
    return downloaded
