# Repository FAQ

## Quick Summary

This repository is a research workspace for a glucose-responsive insulin (GRI) project. It combines:

- A paper in `paper/` with figures, tables, and the compiled PDF.
- A reusable Python package in `packages/gri-hybrid-ranker/` that exposes the hybrid tri-state ranking workflow as a library.
- A reproducible experiment runner in `experiments/gri_hybrid_validation/` that simulates benchmark results, writes artifacts, and feeds the paper outputs.
- Research context in `knowledge/` and `sources/`.

The codebase is primarily a deterministic simulation and packaging project, not a production wet-lab or molecular-dynamics pipeline.

## What is the main purpose of the repo?

The repo packages and documents a computational approach for prioritizing candidate glucose-responsive insulin designs. The central idea is to rank synthetic candidates across low, normal, and high glucose states, then compare the proposed method against several baselines using acceptance criteria tied to safety, monotonicity, stability, and robustness.

## What are the important top-level directories?

- `paper/`: LaTeX source, bibliography, compiled PDF, figures, tables, and exported datasets used by the manuscript.
- `packages/gri-hybrid-ranker/`: Installable Python package with the reusable ranking pipeline and package tests.
- `experiments/gri_hybrid_validation/`: CLI runner, experiment config, experiment implementation, outputs, and tests for the validation simulation.
- `knowledge/`: Literature synthesis and working notes.
- `sources/`: Collected source metadata.

## What is the source of truth for the Python package?

Use `packages/gri-hybrid-ranker/src/gri_hybrid_ranker/` as the source of truth.

`packages/gri-hybrid-ranker/build/lib/` is a build artifact mirror, and `packages/gri-hybrid-ranker/dist/` contains packaged release files. Those should not be treated as the primary implementation when reading or editing code.

## What does the `gri-hybrid-ranker` package do?

The package wraps the reusable parts of the project into a library API:

- `PipelineConfig` and `ExperimentSpec` define the experiment matrix.
- `GRIHybridRanker` generates a deterministic synthetic candidate dataset.
- The pipeline simulates metrics for the proposed method and baselines across seeds and sweep settings.
- It summarizes metrics, evaluates acceptance checks, runs symbolic consistency checks, and writes output artifacts.

The package is centered on synthetic benchmarking rather than training a learned model from raw biological data.

## What does the experiment runner do that the package does not?

The experiment runner in `experiments/gri_hybrid_validation/` is the paper-facing orchestration layer. It:

- Loads the full experiment matrix from `configs/experiments.json`.
- Runs the deterministic simulation across all configured hypotheses.
- Exports tables and datasets into `experiments/.../outputs/`.
- Copies paper inputs into `paper/data/` and `paper/tables/`.
- Renders the final PDF figures into `paper/figures/`.
- Writes a run manifest and summary JSON for downstream reporting.

In short, the package is the reusable core; the experiment directory is the full validation workflow.

## Is this a real ML training pipeline?

No. The current code simulates candidate features and benchmark outcomes with deterministic random number generation. It is useful for:

- demonstrating methodology,
- validating artifact flow,
- checking acceptance logic,
- packaging the workflow,
- and producing paper-ready outputs.

It does not currently ingest real assay data, train a neural model, or run molecular dynamics.

## How are experiments configured?

The experiment matrix lives in `experiments/gri_hybrid_validation/configs/experiments.json`.

It defines:

- experiment IDs,
- referenced datasets,
- baselines,
- seeds,
- metric names,
- hyperparameter sweep grids,
- and acceptance criteria.

The configured experiments cover:

- H1 multistate ranking,
- H3 hybrid ML/MD reranking,
- H4 Pareto tradeoff selection,
- and symbolic or boundary stress checks.

## How do I run the full validation workflow?

From the repo root:

```bash
experiments/.venv/bin/python experiments/gri_hybrid_validation/run_experiments.py \
  --config experiments/gri_hybrid_validation/configs/experiments.json \
  --output-dir experiments/gri_hybrid_validation/outputs
```

This writes fresh outputs under `experiments/gri_hybrid_validation/outputs/` and updates paper-facing artifacts under `paper/`.

## How do I use the package directly?

For local editable development:

```bash
pip install -e packages/gri-hybrid-ranker
```

Then use the API from Python:

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
        )
    ]
)

result = GRIHybridRanker().run(config=config, output_dir=Path("outputs"))
```

## Where do the paper figures and tables come from?

They are generated by the validation experiment pipeline and copied into `paper/`:

- figures: `paper/figures/`
- tables: `paper/tables/`
- datasets: `paper/data/`

This means the manuscript assets are tied directly to the experiment outputs already checked into the repo.

## What do the tests cover?

There are two small test suites:

- `packages/gri-hybrid-ranker/tests/test_pipeline.py` checks that the packaged pipeline writes its expected artifacts.
- `experiments/gri_hybrid_validation/tests/` checks metric summarization and symbolic report generation.

Run them with:

```bash
python -m pytest packages/gri-hybrid-ranker/tests -q
python -m pytest experiments/gri_hybrid_validation/tests -q
```

## Are the checked-in outputs generated or handwritten?

Many files under `experiments/gri_hybrid_validation/outputs/`, `paper/data/`, `paper/tables/`, `paper/figures/`, and `packages/gri-hybrid-ranker/dist/` are generated artifacts. They are useful for inspection and reproducibility, but the code under `src/` and the config JSON are the main editable inputs.

## What dependencies matter?

The package runtime depends on:

- `numpy`
- `pandas`
- `sympy`

Optional plotting and development dependencies are declared in `packages/gri-hybrid-ranker/pyproject.toml`, including `matplotlib`, `seaborn`, `pytest`, `ruff`, and `mypy`.

## Does the repo require GPUs?

Not for the current checked-in implementation. The project metadata mentions a 2-GPU budget for the broader research context, but the present Python code is a lightweight synthetic simulation built on NumPy, Pandas, and SymPy.

## What is the role of symbolic validation here?

Both the package and the experiment workflow include SymPy-based checks that verify simple symbolic properties of the scoring formulation, such as penalty sign conventions and monotonicity-related identities. This is a consistency safeguard for the ranking logic, not a formal proof of biological correctness.

## Are there any caveats when rerunning the symbolic checks?

Yes. The experiment code references `phase_outputs/SYMPY.md` as an external spec path. If that file is missing in your environment, symbolic report generation may still run, but the spec-existence check will reflect that absence. The package-level symbolic checks are otherwise self-contained.

## If I want to modify the methodology, where should I start?

Start with:

- `packages/gri-hybrid-ranker/src/gri_hybrid_ranker/pipeline.py` for package behavior,
- `packages/gri-hybrid-ranker/src/gri_hybrid_ranker/analysis.py` for aggregation and confirmatory summaries,
- `packages/gri-hybrid-ranker/src/gri_hybrid_ranker/symbolic.py` for symbolic checks,
- `experiments/gri_hybrid_validation/configs/experiments.json` for experiment design,
- and `experiments/gri_hybrid_validation/src/gri_validation/core.py` for the paper-generation workflow.

## What is the main limitation of the repository as it stands?

The repository is strong as a reproducible research artifact and packaging exercise, but limited as a biological discovery system because it currently uses synthetic data and simulated metrics rather than experimental measurements or real structure-conditioned model training.
