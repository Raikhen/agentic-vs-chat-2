"""Shared plotting utilities for refusal-rate experiments.

Provides parameterized chart functions that work across experiments with
different condition sets, colors, and styling via a ``style`` dict.

Expected ``style`` keys::

    condition_order:    list[str]
    condition_labels:   dict[str, str]
    condition_colors:   dict[str, str]
    bar_decorator:      Callable[[bars, conditions], None] | None
    aggregate_figsize:  tuple[float, float]
    bar_width:          float
    aggregate_title:    str
    per_model_title:    str
    aggregate_filename: str
    per_model_filename: str
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bootstrap_ci(
    scores: np.ndarray, n_boot: int = 10_000, ci: float = 0.95
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(42)
    boot_means = np.array(
        [scores[rng.integers(0, len(scores), len(scores))].mean() for _ in range(n_boot)]
    )
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_means, 100 * alpha)), float(
        np.percentile(boot_means, 100 * (1 - alpha))
    )


def balanced_df(df: pd.DataFrame, condition_order: list[str]) -> pd.DataFrame:
    """Return balanced subset: for each model, keep only sample IDs present in all conditions."""
    conditions = set(condition_order)
    rows = []
    for _model, group in df.groupby("model"):
        present_conditions = set(group["condition"].unique())
        if not conditions.issubset(present_conditions):
            continue
        # Find sample IDs that appear in every condition
        ids_per_cond = [
            set(group.loc[group["condition"] == c, "sample_id"]) for c in condition_order
        ]
        common_ids = ids_per_cond[0]
        for s in ids_per_cond[1:]:
            common_ids &= s
        if common_ids:
            rows.append(group[group["sample_id"].isin(common_ids)])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_refusal_by_condition(df: pd.DataFrame, out_dir: Path, style: dict[str, Any]):
    """Bar chart of mean refusal score per condition with 95% bootstrap CIs.

    Uses balanced samples (per-model intersection of sample IDs across all
    conditions) to avoid Simpson's Paradox from unbalanced sample sizes.
    """
    condition_order = style["condition_order"]
    df_bal = balanced_df(df, condition_order)

    conditions = [c for c in condition_order if c in df_bal["condition"].unique()]
    means, ci_lo, ci_hi, ns = [], [], [], []

    for cond in conditions:
        scores = df_bal.loc[df_bal["condition"] == cond, "score"].values
        m = scores.mean()
        lo, hi = bootstrap_ci(scores)
        means.append(m)
        ci_lo.append(m - lo)
        ci_hi.append(hi - m)
        ns.append(len(scores))

    labels = [style["condition_labels"].get(c, c) for c in conditions]
    x = np.arange(len(conditions))

    fig, ax = plt.subplots(figsize=style.get("aggregate_figsize", (7, 4.5)))
    bar_colors = [style["condition_colors"][c] for c in conditions]
    bars = ax.bar(
        x,
        means,
        yerr=[ci_lo, ci_hi],
        capsize=5,
        color=bar_colors,
        edgecolor="white",
        width=style.get("bar_width", 0.55),
        error_kw={"linewidth": 1.5, "color": "#333"},
    )

    bar_decorator = style.get("bar_decorator")
    if bar_decorator is not None:
        bar_decorator(bars, conditions)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean Refusal Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(style.get("aggregate_title", "Refusal Rate by Condition (95% CI)"),
                 fontsize=12, pad=10)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate sample sizes
    for bar, n in zip(bars, ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.02,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="white",
            fontweight="bold",
        )

    fig.tight_layout()
    path = out_dir / style.get("aggregate_filename", "refusal_by_condition.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_refusal_per_model(df: pd.DataFrame, out_dir: Path, style: dict[str, Any]):
    """One subplot per model showing refusal by condition with 95% bootstrap CIs."""
    condition_order = style["condition_order"]
    conditions = [c for c in condition_order if c in df["condition"].unique()]
    models = sorted(df["model"].unique())

    def short_name(m: str) -> str:
        return m.split("/")[-1].split(":")[-1]

    n_models = len(models)
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    bar_decorator = style.get("bar_decorator")

    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        model_df = df[df["model"] == model]

        means, ci_lo, ci_hi, ns = [], [], [], []
        present = []
        for cond in conditions:
            scores = model_df.loc[model_df["condition"] == cond, "score"].values
            if len(scores) == 0:
                continue
            m = scores.mean()
            lo, hi = bootstrap_ci(scores)
            means.append(m)
            ci_lo.append(m - lo)
            ci_hi.append(hi - m)
            ns.append(len(scores))
            present.append(cond)

        x = np.arange(len(present))
        labels = [style["condition_labels"].get(c, c) for c in present]

        bar_colors = [style["condition_colors"][c] for c in present]
        bars = ax.bar(
            x,
            means,
            yerr=[ci_lo, ci_hi],
            capsize=5,
            color=bar_colors,
            edgecolor="white",
            width=style.get("bar_width", 0.55),
            error_kw={"linewidth": 1.5, "color": "#333"},
        )

        if bar_decorator is not None:
            bar_decorator(bars, present)

        for bar, n in zip(bars, ns):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.02,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(short_name(model), fontsize=12, pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        if idx % ncols == 0:
            ax.set_ylabel("Mean Refusal Score", fontsize=10)

    # Hide unused subplots
    for idx in range(n_models, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        style.get("per_model_title", "Refusal Rate by Condition per Model (95% CI)"),
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    path = out_dir / style.get("per_model_filename", "refusal_by_condition_per_model.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
