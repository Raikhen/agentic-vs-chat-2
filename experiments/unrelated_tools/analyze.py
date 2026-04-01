"""Post-run analysis: aggregate results and subgroup breakdowns."""

import argparse

from shared.analysis import analyze, load_logs

CONDITION_ORDER = [
    "bare_chat",
    "system_prompt_only",
    "tools_only",
    "system_prompt_and_tools",
]


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/unrelated_tools",
        help="Directory containing eval logs",
    )
    args = parser.parse_args()

    df, errors_df = load_logs(args.log_dir)
    analyze(df, errors_df, condition_order=CONDITION_ORDER)


if __name__ == "__main__":
    main()
