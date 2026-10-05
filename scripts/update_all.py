"""Full monthly refresh: download data -> rebuild database -> retrain -> forecast -> export dashboard.

    python scripts/update_all.py
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
STEPS = [
    [PY, "scripts/01_collect_data.py"],
    [PY, "scripts/02_build_database.py"],
    [PY, "-m", "measles_predict", "train"],
    [PY, "-m", "measles_predict", "evaluate", "--rolling"],
    [PY, "-m", "measles_predict", "forecast"],
    [PY, "scripts/export_pages.py"],
]

for cmd in STEPS:
    print("\n>>>", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)
print("\nMise à jour terminée.")
