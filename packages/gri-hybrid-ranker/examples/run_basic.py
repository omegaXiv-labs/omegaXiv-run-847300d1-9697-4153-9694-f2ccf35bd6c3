from __future__ import annotations

from pathlib import Path

from gri_hybrid_ranker import ExperimentSpec, GRIHybridRanker, PipelineConfig

config = PipelineConfig(
    experiments=[
        ExperimentSpec(
            id="exp_h1_multistate_ranker_benchmark",
            baselines=[
                "static_single_state_affinity_ranker_gnina_style",
                "tri_state_cross_entropy_classifier_no_pairwise_ranking",
            ],
            seeds=[11, 23],
            sweep_params={"alpha_mix": ["0.2", "0.5"], "shortlist_N": ["100", "200"]},
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
result = ranker.run(config=config, output_dir=Path("./example_outputs"))
print("Acceptance checks:", result["acceptance"])
print("Results summary path:", result["artifacts"].results_summary_json)
