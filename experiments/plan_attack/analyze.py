"""Post-run analysis for the plan-attack experiment."""

import argparse

from shared.analysis import analyze, load_logs

CONDITION_ORDER = ["raw", "plan"]


def main():
    parser = argparse.ArgumentParser(
        description="Analyze plan-attack experiment results"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/plan_attack",
        help="Directory containing eval logs",
    )
    args = parser.parse_args()

    df, errors_df = load_logs(args.log_dir)
    analyze(df, errors_df, condition_order=CONDITION_ORDER)


if __name__ == "__main__":
    main()
