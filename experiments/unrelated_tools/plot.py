"""Generate figures from evaluation results."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shared.analysis import load_logs
from shared.plotting import bootstrap_ci, plot_refusal_by_condition, plot_refusal_per_model

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

CONDITION_COLORS = {
    "bare_chat": "#8C8C8C",            # neutral gray (baseline)
    "system_prompt_only": "#4A90D9",    # blue (structure)
    "tools_only": "#E8913A",            # amber (capability)
    "system_prompt_and_tools": "#E8913A",  # orange base with blue stripes
}

COMBINED_CONDITION = "system_prompt_and_tools"

plt.rcParams["hatch.linewidth"] = 4.5


def _apply_stripes(bars, conditions):
    """Add blue diagonal stripes to the combined condition bar."""
    for bar, cond in zip(bars, conditions):
        if cond == COMBINED_CONDITION:
            bar.set_hatch("//")
            bar.set_edgecolor("#4A90D9")
            bar.set_linewidth(0.5)


STYLE = {
    "condition_order": CONDITION_ORDER,
    "condition_labels": CONDITION_LABELS,
    "condition_colors": CONDITION_COLORS,
    "bar_decorator": _apply_stripes,
    "aggregate_figsize": (7, 4.5),
    "bar_width": 0.55,
    "aggregate_title": "Refusal Rate by Condition (95% CI)\nBalanced samples across all models",
    "per_model_title": "Refusal Rate by Condition per Model (95% CI)",
    "aggregate_filename": "refusal_by_condition.png",
    "per_model_filename": "refusal_by_condition_per_model.png",
}


# ── Unrelated-tools-only: grouped point plot ─────────────────────────


def plot_refusal_by_condition_and_model(df, out_dir: Path):
    """Grouped point plot: refusal score per condition, broken out by model."""
    conditions = [c for c in CONDITION_ORDER if c in df["condition"].unique()]
    models = sorted(df["model"].unique())

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


def main():
    parser = argparse.ArgumentParser(description="Generate analysis figures")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/unrelated_tools",
        help="Directory containing eval logs",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="figures/unrelated_tools",
        help="Directory to save figures",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, _errors_df = load_logs(args.log_dir)
    if df.empty:
        print("No data to plot.")
        return

    n_conditions = df["condition"].nunique()
    n_models = df["model"].nunique()
    n_samples = len(df)
    print(f"Loaded {n_samples} samples ({n_models} models, {n_conditions} conditions)")

    plot_refusal_by_condition(df, out_dir, STYLE)
    plot_refusal_per_model(df, out_dir, STYLE)
    plot_refusal_by_condition_and_model(df, out_dir)


if __name__ == "__main__":
    main()
