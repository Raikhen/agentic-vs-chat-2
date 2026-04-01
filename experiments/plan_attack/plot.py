"""Generate figures for the plan-attack experiment."""

import argparse
from pathlib import Path

from shared.analysis import load_logs
from shared.plotting import plot_refusal_by_condition, plot_refusal_per_model

STYLE = {
    "condition_order": ["raw", "plan"],
    "condition_labels": {
        "raw": "Raw Prompt",
        "plan": "Plan Prompt",
    },
    "condition_colors": {
        "raw": "#8C8C8C",   # neutral gray (baseline)
        "plan": "#D94A7A",  # magenta-pink (plan)
    },
    "bar_decorator": None,
    "aggregate_figsize": (5, 4.5),
    "bar_width": 0.45,
    "aggregate_title": (
        "Refusal Rate: Raw vs Plan Prompts (95% CI)\n"
        "Balanced samples across all models"
    ),
    "per_model_title": "Refusal Rate: Raw vs Plan per Model (95% CI)",
    "aggregate_filename": "refusal_raw_vs_plan.png",
    "per_model_filename": "refusal_raw_vs_plan_per_model.png",
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate plan-attack experiment figures"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/plan_attack",
        help="Directory containing eval logs",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="figures/plan_attack",
        help="Directory to save figures",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, _ = load_logs(args.log_dir)
    if df.empty:
        print("No data to plot.")
        return

    n_conditions = df["condition"].nunique()
    n_models = df["model"].nunique()
    n_samples = len(df)
    print(f"Loaded {n_samples} samples ({n_models} models, {n_conditions} conditions)")

    plot_refusal_by_condition(df, out_dir, STYLE)
    plot_refusal_per_model(df, out_dir, STYLE)


if __name__ == "__main__":
    main()
