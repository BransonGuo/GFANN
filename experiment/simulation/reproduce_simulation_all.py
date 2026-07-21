from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run(script_name: str) -> None:
    print(f"\n=== Running {script_name} ===")
    subprocess.check_call([sys.executable, str(SCRIPT_DIR / script_name)])


if __name__ == "__main__":
    for script in [
        "reproduce_simulation_main.py",
        "reproduce_simulation_ablation.py",
        "reproduce_simulation_more_nonlinear.py",
        "reproduce_simulation_joint_freeze.py",
    ]:
        run(script)
