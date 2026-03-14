from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _setup_theme() -> None:
    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")


def _plot_metric(ax, sub: pd.DataFrame, title: str, y_label: str) -> None:
    sub = sub.copy().sort_values("mean", ascending=False).head(4)
    x = range(len(sub))
    ax.bar(
        x,
        sub["mean"],
        yerr=[sub["mean"] - sub["ci95_low"], sub["ci95_high"] - sub["mean"]],
        capsize=4,
        label="Mean ±95% CI",
    )
    ax.set_xticks(list(x), sub["method"], rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Method")
    ax.legend(loc="best")


def plot_main_panels(summary_df: pd.DataFrame, out_path: Path) -> None:
    _setup_theme()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    h1 = summary_df[(summary_df["experiment_id"] == "exp_h1_multistate_ranker_benchmark") & (summary_df["metric"] == "spearman_delta_rank_low_high")]
    h3 = summary_df[(summary_df["experiment_id"] == "exp_h3_hybrid_ml_md_reranking") & (summary_df["metric"] == "false_positive_rate_top30")]
    h4 = summary_df[(summary_df["experiment_id"] == "exp_h4_pareto_selection_tradeoff") & (summary_df["metric"] == "nondominated_ratio_topK")]
    hs = summary_df[(summary_df["experiment_id"] == "exp_hybrid_symbolic_assumption_boundary_counterexample") & (summary_df["metric"] == "boundary_case_ordering_pass_rate")]

    _plot_metric(axes[0, 0], h1, "H1: Tri-State Rank Correlation", "Spearman delta rank")
    _plot_metric(axes[0, 1], h3, "H3: Low-Glucose False Positive Rate", "FPR (top-30)")
    _plot_metric(axes[1, 0], h4, "H4: Pareto Nondominated Ratio", "Nondominated ratio (top-K)")
    _plot_metric(axes[1, 1], hs, "Stress: Boundary Ordering Pass", "Pass rate")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_confirmatory_panels(confirm_df: pd.DataFrame, out_path: Path) -> None:
    _setup_theme()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    sns.barplot(data=confirm_df, x="regime", y="regime_precision", ax=axes[0], label="Precision")
    axes[0].errorbar(
        range(len(confirm_df)),
        confirm_df["regime_precision"],
        yerr=[confirm_df["regime_precision"] - confirm_df["ci95_low"], confirm_df["ci95_high"] - confirm_df["regime_precision"]],
        fmt="none",
        ecolor="black",
        capsize=4,
        label="95% CI",
    )
    axes[0].set_title("Confirmatory Regime-Stratified Precision")
    axes[0].set_xlabel("Glycemic response regime")
    axes[0].set_ylabel("Precision (unitless)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(loc="best")

    sns.lineplot(data=confirm_df, x="regime", y="tradeoff_gap", marker="o", ax=axes[1], label="Trade-off gap")
    axes[1].set_title("Robustness-Performance Trade-off by Regime")
    axes[1].set_xlabel("Glycemic response regime")
    axes[1].set_ylabel("Trade-off gap (unitless)")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
