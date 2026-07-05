

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    sync = Path(__file__).resolve().parent / "sync_labeled_dataset.py"
    raise SystemExit(subprocess.call([sys.executable, str(sync)] + sys.argv[1:]))
