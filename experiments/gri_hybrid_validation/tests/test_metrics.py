from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gri_validation.analysis import summarize_metrics  # noqa: E402


def test_summarize_metrics_basic() -> None:
    df = pd.DataFrame(
        [
            {"experiment_id": "e1", "method": "m1", "metric": "x", "value": 0.1},
            {"experiment_id": "e1", "method": "m1", "metric": "x", "value": 0.3},
            {"experiment_id": "e1", "method": "m2", "metric": "x", "value": 0.2},
        ]
    )
    out = summarize_metrics(df)
    row = out[(out["experiment_id"] == "e1") & (out["method"] == "m1") & (out["metric"] == "x")].iloc[0]
    assert abs(float(row["mean"]) - 0.2) < 1e-9
    assert row["ci95_high"] >= row["ci95_low"]
