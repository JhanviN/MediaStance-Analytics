"""
Sets up Windows Task Scheduler to run the live pipeline automatically.

Usage:
    python scripts/setup_scheduler.py --interval 60   # every 60 minutes
    python scripts/setup_scheduler.py --remove        # remove the task

The task runs live_pipeline.py --once every N minutes.
Logs are saved to logs/scheduler.log
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_NAME = "MediaStanceAnalytics_LivePipeline"
LOG_DIR = ROOT / "logs"


def create_task(interval_minutes: int = 60) -> None:
    LOG_DIR.mkdir(exist_ok=True)

    python_exe = sys.executable
    script = ROOT / "scripts" / "live_pipeline.py"
    log_file = LOG_DIR / "scheduler.log"

    # Command that runs the pipeline and appends to log
    cmd = f'"{python_exe}" "{script}" --once --model baseline >> "{log_file}" 2>&1'

    # Create a wrapper batch file (Task Scheduler works better with .bat)
    bat_file = ROOT / "run_pipeline_scheduled.bat"
    bat_file.write_text(
        f'@echo off\n'
        f'cd /d "{ROOT}"\n'
        f'echo [%date% %time%] Running MediaStance live pipeline >> "{log_file}"\n'
        f'"{python_exe}" "{script}" --once --model baseline >> "{log_file}" 2>&1\n'
        f'echo [%date% %time%] Done >> "{log_file}"\n',
        encoding="utf-8"
    )
    print(f"Created: {bat_file}")

    # Register with Windows Task Scheduler
    # Runs every N minutes, starting now
    schtasks_cmd = [
        "schtasks", "/create",
        "/tn", TASK_NAME,
        "/tr", str(bat_file),
        "/sc", "MINUTE",
        "/mo", str(interval_minutes),
        "/f",  # force overwrite if exists
        "/rl", "HIGHEST",  # run with highest privileges
    ]

    print(f"\nRegistering Windows Task Scheduler task: {TASK_NAME}")
    print(f"Interval: every {interval_minutes} minutes")
    print(f"Log: {log_file}")

    result = subprocess.run(schtasks_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"\nTask created successfully.")
        print(f"To verify: schtasks /query /tn {TASK_NAME}")
        print(f"To run now: schtasks /run /tn {TASK_NAME}")
        print(f"To remove: python scripts/setup_scheduler.py --remove")
    else:
        print(f"\nFailed to create task: {result.stderr}")
        print(f"\nManual setup instructions:")
        print(f"1. Open Task Scheduler (search in Start menu)")
        print(f"2. Create Basic Task → name: {TASK_NAME}")
        print(f"3. Trigger: Daily, repeat every {interval_minutes} minutes")
        print(f"4. Action: Start a program → {bat_file}")


def remove_task() -> None:
    result = subprocess.run(
        ["schtasks", "/delete", "/tn", TASK_NAME, "/f"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"Task '{TASK_NAME}' removed.")
    else:
        print(f"Could not remove task: {result.stderr}")

    # Clean up bat file
    bat_file = ROOT / "run_pipeline_scheduled.bat"
    if bat_file.exists():
        bat_file.unlink()
        print(f"Removed {bat_file.name}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=60,
                    help="Minutes between runs (default 60)")
    ap.add_argument("--remove", action="store_true",
                    help="Remove the scheduled task")
    args = ap.parse_args()

    if args.remove:
        remove_task()
    else:
        create_task(args.interval)


if __name__ == "__main__":
    main()
