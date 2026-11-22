import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.research_benchmark import (
    ExperimentRun,
    RESEARCH_RESULTS_PATH,
    save_comparison_charts,
)


def main() -> int:
    if not RESEARCH_RESULTS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing benchmark output: {RESEARCH_RESULTS_PATH}"
        )
    history = json.loads(RESEARCH_RESULTS_PATH.read_text(encoding="utf-8"))
    experiments = history["experiments"]
    if len(experiments) < 2:
        raise ValueError("At least one clean and one injected run are required")
    latest_two = [
        ExperimentRun.model_validate(run)
        for run in experiments[-2:]
    ]
    save_comparison_charts(latest_two)
    print("Regenerated Vietnamese accuracy, manipulation and effectiveness PNG charts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
