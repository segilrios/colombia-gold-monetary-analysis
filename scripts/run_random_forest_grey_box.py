from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gold_policy_analysis.supervised import run_random_forest_grey_box


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Random Forest and grey-box Random Forest models."
    )
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args()

    outputs = run_random_forest_grey_box(args.config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
