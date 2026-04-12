#!/usr/bin/env python3
"""
Backward-compatible entry point — delegates to sync_labeled_dataset.py

Use either:
  python scripts/init_labeled_template.py --latest
  python scripts/sync_labeled_dataset.py --latest
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    sync = Path(__file__).resolve().parent / "sync_labeled_dataset.py"
    raise SystemExit(subprocess.call([sys.executable, str(sync)] + sys.argv[1:]))
