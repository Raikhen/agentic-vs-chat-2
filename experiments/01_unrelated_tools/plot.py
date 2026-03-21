"""Generate figures from evaluation results."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import load_logs

CONDITION_ORDER = [
    "bare_chat",
    "system_prompt_only",
    "tools_only",
    "system_prompt_and_tools",
]

CONDITION_LABELS = {
    "bare_chat": "Bare Chat",
    "system_prompt_only": "System Prompt",
    "tools_only": "Tools",
    "system_prompt_and_tools": "System Prompt\n+ Tools",
}


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


def plot_refusal_by_condition(df: pd.DataFrame, out_dir: Path):
    """Bar chart of mean refusal score per condition with 95% bootstrap CIs."""
    conditions = [c for c in CONDITION_ORDER if c in df["condition"].unique()]
    means, ci_lo, ci_hi, ns = [], [], [], []

    for cond in conditions:
        scores = df.loc[df["condition"] == cond, "score"].values
        m = scores.mean()
        lo, hi = bootstrap_ci(scores)
        means.append(m)
        ci_lo.append(m - lo)
        ci_hi.append(hi - m)
        ns.append(len(scores))

    labels = [CONDITION_LABELS.get(c, c) for c in conditions]
    x = np.arange(len(conditions))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        x,
        means,
        yerr=[ci_lo, ci_hi],
        capsize=5,
        color="#5B8DB8",
        edgecolor="white",
        width=0.55,
        error_kw={"linewidth": 1.5, "color": "#333"},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean Refusal Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Refusal Rate by Condition (95% CI)", fontsize=13, pad=10)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate sample sizes
    for i, (bar, n) in enumerate(zip(bars, ns)):
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
    path = out_dir / "refusal_by_condition.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_refusal_by_condition_and_model(df: pd.DataFrame, out_dir: Path):
    """Grouped point plot: refusal score per condition, broken out by model."""
    conditions = [c for c in CONDITION_ORDER if c in df["condition"].unique()]
    models = sorted(df["model"].unique())

    # Shorten model names for display
    def short_name(m: str) -> str:
        return m.split("/")[-1].split(":")[-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    n_models = len(models)
    offsets = np.linspace(-0.25, 0.25, n_models) if n_models > 1 else [0.0]
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_models, 3)))

    for j, model in enumerate(models):
        ms, los, his, xs = [], [], [], []
        for i, cond in enumerate(conditions):
            scores = df.loc[
                (df["condition"] == cond) & (df["model"] == model), "score"
            ].values
            if len(scores) == 0:
                continue
            m = scores.mean()
            lo, hi = bootstrap_ci(scores)
            ms.append(m)
            los.append(m - lo)
            his.append(hi - m)
            xs.append(i + offsets[j])

        ax.errorbar(
            xs,
            ms,
            yerr=[los, his],
            fmt="o",
            label=short_name(model),
            color=colors[j],
            capsize=3,
            markersize=6,
            linewidth=1.2,
        )

    labels = [CONDITION_LABELS.get(c, c) for c in conditions]
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean Refusal Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Refusal Rate by Condition and Model (95% CI)", fontsize=13, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)

    fig.tight_layout()
    path = out_dir / "refusal_by_condition_and_model.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_refusal_by_condition_per_model(df: pd.DataFrame, out_dir: Path):
    """One subplot per model showing refusal by condition with 95% bootstrap CIs."""
    conditions = [c for c in CONDITION_ORDER if c in df["condition"].unique()]
    models = sorted(df["model"].unique())

    def short_name(m: str) -> str:
        return m.split("/")[-1].split(":")[-1]

    n_models = len(models)
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

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
        labels = [CONDITION_LABELS.get(c, c) for c in present]

        bars = ax.bar(
            x,
            means,
            yerr=[ci_lo, ci_hi],
            capsize=5,
            color="#5B8DB8",
            edgecolor="white",
            width=0.55,
            error_kw={"linewidth": 1.5, "color": "#333"},
        )

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

    fig.suptitle("Refusal Rate by Condition per Model (95% CI)", fontsize=14, y=1.01)
    fig.tight_layout()
    path = out_dir / "refusal_by_condition_per_model.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate analysis figures")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/01_unrelated_tools",
        help="Directory containing eval logs",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="figures/01_unrelated_tools",
        help="Directory to save figures",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_logs(args.log_dir)
    if df.empty:
        print("No data to plot.")
        return

    n_conditions = df["condition"].nunique()
    n_models = df["model"].nunique()
    n_samples = len(df)
    print(f"Loaded {n_samples} samples ({n_models} models, {n_conditions} conditions)")

    plot_refusal_by_condition(df, out_dir)
    plot_refusal_by_condition_and_model(df, out_dir)
    plot_refusal_by_condition_per_model(df, out_dir)


if __name__ == "__main__":
    main()
