from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure local src package import when invoked from workspace root.
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gri_validation.core import ProgressReporter, run_pipeline  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRI hybrid validation simulation experiments")
    parser.add_argument("--config", default=str(THIS_DIR / "configs" / "experiments.json"))
    parser.add_argument("--output-dir", default=str(THIS_DIR / "outputs"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    project_root = THIS_DIR.parent.parent.resolve()

    sink = os.environ.get("QUARKS_PROGRESS_EVENT_SINK") or os.environ.get("QUARKS_RUN_PROGRESS_EVENTS_PATH")
    reporter = ProgressReporter(sink)
    reporter.start_task("validation_simulation", "validation_simulation")
    print("progress: 5% | loading config")

    result = run_pipeline(config_path=config_path, output_dir=output_dir, project_root=project_root, reporter=reporter)

    summary_path = output_dir / "run_manifest.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "results_summary_path": str(result["results_summary_path"]),
                "figure_paths": [str(p) for p in result["figure_paths"]],
                "table_paths": [str(p) for p in result["table_paths"]],
                "dataset_paths": [str(p) for p in result["dataset_paths"]],
                "sympy_report_path": str(result["sympy_report_path"]),
            },
            f,
            indent=2,
        )

    reporter.finish("validation_simulation", "completed")
    print("progress: 100% | validation simulation completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
