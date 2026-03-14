from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_metrics(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metric means and confidence intervals across seeds/sweeps."""
    grouped = raw_df.groupby(["experiment_id", "method", "metric"], as_index=False).agg(
        mean=("value", "mean"),
        std=("value", "std"),
        n=("value", "count"),
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["se"] = grouped["std"] / np.sqrt(grouped["n"].clip(lower=1))
    grouped["ci95_low"] = grouped["mean"] - 1.96 * grouped["se"]
    grouped["ci95_high"] = grouped["mean"] + 1.96 * grouped["se"]
    return grouped.sort_values(["experiment_id", "metric", "method"]).reset_index(drop=True)


def _bootstrap_mean(
    values: np.ndarray,
    n_boot: int = 1000,
    seed: int = 20260314,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    samples = []
    n_values = len(values)
    for _ in range(n_boot):
        indices = rng.integers(0, n_values, n_values)
        samples.append(float(np.mean(values[indices])))
    boot = np.array(samples)
    return float(np.mean(values)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def confirmatory_regime_analysis(base_df: pd.DataFrame) -> pd.DataFrame:
    """Compute regime-level precision/tradeoff summaries from tri-state candidate signals."""
    quantiles = np.quantile(
        base_df["activity_high"] - base_df["activity_low"],
        [0.2, 0.4, 0.6, 0.8],
    )

    def regime(delta: float) -> str:
        if delta <= quantiles[0]:
            return "low_response"
        if delta <= quantiles[1]:
            return "midlow_response"
        if delta <= quantiles[2]:
            return "midhigh_response"
        if delta <= quantiles[3]:
            return "high_response"
        return "very_high_response"

    enriched = base_df.copy()
    enriched["delta"] = enriched["activity_high"] - enriched["activity_low"]
    enriched["regime"] = enriched["delta"].apply(regime)

    rows: list[dict[str, float | str | int]] = []
    for rg, part in enriched.groupby("regime"):
        precision_proxy = np.clip(
            part["activity_high"] - 0.7 * part["low_glucose_risk"] - 0.2 * part["uncertainty"],
            0.0,
            1.0,
        )
        mean_value, low_ci, high_ci = _bootstrap_mean(precision_proxy.to_numpy())
        rows.append(
            {
                "regime": rg,
                "regime_precision": mean_value,
                "ci95_low": low_ci,
                "ci95_high": high_ci,
                "n_candidates": int(len(part)),
                "stress_strength": "strong" if "low" in rg else "moderate",
                "tradeoff_gap": float(max(0.0, 0.17 - 0.02 * mean_value)),
            }
        )

    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)
