from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentSpec:
    """Experiment recipe used by the hybrid ranker simulator."""

    id: str
    baselines: list[str]
    seeds: list[int]
    sweep_params: dict[str, list[str]]

    @classmethod
    def from_dict(cls, payload: dict) -> ExperimentSpec:
        return cls(
            id=str(payload["id"]),
            baselines=[str(v) for v in payload.get("baselines", [])],
            seeds=[int(v) for v in payload.get("seeds", [])],
            sweep_params={
                str(k): [str(x) for x in vals]
                for k, vals in payload.get("sweep_params", {}).items()
            },
        )


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level package configuration for hybrid ranker execution."""

    experiments: list[ExperimentSpec]

    @classmethod
    def from_dict(cls, payload: dict) -> PipelineConfig:
        experiments = [ExperimentSpec.from_dict(exp) for exp in payload.get("experiments", [])]
        if not experiments:
            raise ValueError("PipelineConfig requires at least one experiment definition")
        return cls(experiments=experiments)


@dataclass(frozen=True)
class PipelineArtifacts:
    """Resolved artifact paths written by a pipeline run."""

    raw_metrics_csv: Path
    metrics_summary_csv: Path
    acceptance_checks_csv: Path
    confirmatory_regime_csv: Path
    sympy_validation_json: Path
    results_summary_json: Path
