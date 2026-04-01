"""Generate figures for the competitive-pressure ablation experiment."""

import argparse
from pathlib import Path

from shared.analysis import load_logs
from shared.plotting import plot_refusal_by_condition, plot_refusal_per_model

STYLE = {
    "condition_order": ["no_suffix", "neutral_suffix", "competition"],
    "condition_labels": {
        "no_suffix": "No Suffix",
        "neutral_suffix": "Neutral\nEvaluation",
        "competition": "Competitive\nFraming",
    },
    "condition_colors": {
        "no_suffix": "#8C8C8C",       # neutral gray (baseline)
        "neutral_suffix": "#4A90D9",   # blue (neutral eval)
        "competition": "#D94A4A",      # red (competitive pressure)
    },
    "bar_decorator": None,
    "aggregate_figsize": (6, 4.5),
    "bar_width": 0.50,
    "aggregate_title": (
        "Refusal Rate by Competitive Framing (95% CI)\n"
        "Balanced samples across all models"
    ),
    "per_model_title": "Refusal Rate by Competitive Framing per Model (95% CI)",
    "aggregate_filename": "refusal_by_condition.png",
    "per_model_filename": "refusal_by_condition_per_model.png",
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate competitive-pressure experiment figures"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/competitive_pressure",
        help="Directory containing eval logs",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="figures/competitive_pressure",
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
