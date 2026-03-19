from __future__ import annotations

from pathlib import Path

from gri_hybrid_ranker import ExperimentSpec, GRIHybridRanker, PipelineConfig


def test_pipeline_writes_expected_artifacts(tmp_path: Path) -> None:
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

    ranker = GRIHybridRanker(max_sweep_points=3)
    result = ranker.run(config=config, output_dir=tmp_path)

    artifacts = result["artifacts"]
    assert artifacts.raw_metrics_csv.exists()
    assert artifacts.metrics_summary_csv.exists()
    assert artifacts.acceptance_checks_csv.exists()
    assert artifacts.confirmatory_regime_csv.exists()
    assert artifacts.sympy_validation_json.exists()
    assert artifacts.results_summary_json.exists()
    assert "h1_spearman_gain" in result["acceptance"]
