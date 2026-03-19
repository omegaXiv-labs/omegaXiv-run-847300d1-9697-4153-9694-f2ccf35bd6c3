# GRI Hybrid Validation Simulation

This experiment package implements the `validation_simulation` phase for the selected path
`path_p3_hybrid_ml_md_pareto`, aligned to hypotheses H1/H3/H4 and symbolic checks in
`phase_outputs/SYMPY.md`.

## Structure

- `run_experiments.py`: thin CLI entrypoint.
- `configs/experiments.json`: experiment matrix (datasets, baselines, seeds, sweeps, metrics, criteria).
- `src/gri_validation/core.py`: deterministic simulation pipeline and artifact generation.
- `src/gri_validation/analysis.py`: metric aggregation, confidence intervals, confirmatory checks.
- `src/gri_validation/plotting.py`: seaborn-styled multi-panel PDF figure export.
- `src/gri_validation/sympy_checks.py`: symbolic validation checks and report export.
- `tests/*.py`: reproducibility and symbolic-validation tests.

## Reproducible Commands

From workspace root:

```bash
experiments/.venv/bin/python experiments/gri_hybrid_validation/run_experiments.py \
  --config experiments/gri_hybrid_validation/configs/experiments.json \
  --output-dir experiments/gri_hybrid_validation/outputs
```

Then run quality checks:

```bash
experiments/.venv/bin/python -m pytest experiments/gri_hybrid_validation/tests -q
```

## Outputs

- Core outputs: `experiments/gri_hybrid_validation/outputs/`
- Figures (PDF): `paper/figures/`
- Tables/Data exports: `paper/tables/`, `paper/data/`
- Results summary + caption handoff: `experiments/gri_hybrid_validation/outputs/results_summary.json`
