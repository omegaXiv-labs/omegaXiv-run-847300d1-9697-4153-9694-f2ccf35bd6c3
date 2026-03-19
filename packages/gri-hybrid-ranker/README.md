# gri-hybrid-ranker

## Overview

`gri-hybrid-ranker` is a reusable Python library that packages omegaXiv's core hybrid tri-state ranking methodology for glucose-responsive insulin candidate prioritization. It extracts and productizes the contribution implemented in the project validation phase: deterministic multi-experiment simulation, acceptance-gate evaluation, regime-level confirmatory analysis, and symbolic consistency checks.

## Installation

Canonical user flow:

1. `pip install omegaxiv`
2. `ox install gri-hybrid-ranker==0.1.0`

Maintainer/dev-only source install from this repository subdirectory:

```bash
pip install -e packages/gri-hybrid-ranker
```

## Configuration

The API accepts a `PipelineConfig` containing one or more `ExperimentSpec` entries:

- `id`: experiment identifier (`exp_h1_multistate_ranker_benchmark`, etc.)
- `baselines`: baseline method names used for comparison
- `seeds`: deterministic seeds
- `sweep_params`: discrete hyperparameter grids sampled by the runner

## Usage Examples

Basic programmatic usage:

```python
from pathlib import Path
from gri_hybrid_ranker import ExperimentSpec, GRIHybridRanker, PipelineConfig

config = PipelineConfig(
    experiments=[
        ExperimentSpec(
            id="exp_h1_multistate_ranker_benchmark",
            baselines=["static_single_state_affinity_ranker_gnina_style"],
            seeds=[11, 23],
            sweep_params={"alpha_mix": ["0.2", "0.5"]},
        ),
        ExperimentSpec(
            id="exp_hybrid_symbolic_assumption_boundary_counterexample",
            baselines=["adversarial_randomized_scoring_policy"],
            seeds=[11, 23],
            sweep_params={"counterexample_strength": ["mild", "strong"]},
        ),
    ]
)

ranker = GRIHybridRanker(max_sweep_points=4)
result = ranker.run(config=config, output_dir=Path("./outputs"))
print(result["acceptance"])
print(result["artifacts"].results_summary_json)
```

Run the included example script:

```bash
python packages/gri-hybrid-ranker/examples/run_basic.py
```

## Troubleshooting

- `ValueError: PipelineConfig requires at least one experiment definition`
  - Ensure your config includes at least one `ExperimentSpec`.
- `ModuleNotFoundError: sympy`
  - Install dependencies from `requirements.txt` in the package root.
- Acceptance checks seem unexpectedly strict
  - This package preserves the original project acceptance thresholds; adjust baseline selection and sweep coverage instead of changing gate logic unless you are intentionally forking methodology.
