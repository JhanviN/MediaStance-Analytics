"""
Upload the trained transformer model to Hugging Face Hub.
Run this once before deploying to HF Spaces.

Usage:
    python scripts/upload_model_to_hub.py --repo YOUR_HF_USERNAME/mediastance-model
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "transformer_bilateral"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True,
                    help="HF Hub repo id e.g. JhanviN/mediastance-model")
    ap.add_argument("--token", default=None,
                    help="HF token (or set HF_TOKEN env var)")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Run: pip install huggingface_hub")
        sys.exit(1)

    import os
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Set HF_TOKEN environment variable or pass --token")
        sys.exit(1)

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True)
        print(f"Repo: https://huggingface.co/{args.repo}")
    except Exception as e:
        print(f"Repo creation: {e}")

    # Upload model files
    print(f"Uploading from {MODEL_DIR}...")
    api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=args.repo,
        repo_type="model",
        ignore_patterns=["checkpoint-*", "optimizer.pt", "rng_state.pth", "scheduler.pt"],
    )
    print(f"\nModel uploaded to: https://huggingface.co/{args.repo}")
    print(f"\nAdd this to your GitHub Actions secrets:")
    print(f"  HF_TOKEN = your_hf_token")
    print(f"  HF_MODEL_REPO = {args.repo}")  # JhanviN/mediastance-deploy


if __name__ == "__main__":
    main()
