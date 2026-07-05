# HuggingFace Spaces entry point — redirects to the Streamlit dashboard
# HF Spaces looks for app.py at root by default

import subprocess
import sys
from pathlib import Path

# Run the Streamlit app
subprocess.run([
    sys.executable, "-m", "streamlit", "run",
    "main/main_app.py",
    "--server.port", "7860",
    "--server.address", "0.0.0.0",
])
