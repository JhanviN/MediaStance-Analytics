"""
One-time script to upload predictions.db to HF Dataset repo.
Run from repo root: python scripts/upload_db_to_hub.py
"""
from huggingface_hub import HfApi
from pathlib import Path
import os

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

TOKEN = os.environ.get("HF_TOKEN")
DATASET_REPO = os.environ.get("HF_DATASET_REPO", "JhanviN/mediastance-db")
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "predictions.db"

if not TOKEN:
    raise ValueError("HF_TOKEN not set — add it to .env or set it as an env var")

if not DB_PATH.exists():
    raise FileNotFoundError(f"DB not found at {DB_PATH}")

size_mb = DB_PATH.stat().st_size / 1024 / 1024
print(f"Uploading {DB_PATH.name} ({size_mb:.1f} MB) → {DATASET_REPO} ...")

api = HfApi(token=TOKEN)
api.upload_file(
    path_or_fileobj=str(DB_PATH),
    path_in_repo="predictions.db",
    repo_id=DATASET_REPO,
    repo_type="dataset",
    commit_message="upload: predictions.db initial seed",
)

print(f"Done! View at: https://huggingface.co/datasets/{DATASET_REPO}")
