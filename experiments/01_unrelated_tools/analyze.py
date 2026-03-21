"""Post-run analysis: aggregate results and subgroup breakdowns."""

import argparse

import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log
from tabulate import tabulate


def load_logs(log_dir: str) -> pd.DataFrame:
    """Load all eval logs from a directory into a DataFrame."""
    log_refs = list_eval_logs(log_dir)
    rows = []

    for ref in log_refs:
        log = read_eval_log(ref.name)
        model = log.eval.model
        condition = log.eval.task

        if not log.samples:
            continue

        for sample in log.samples:
            if sample.score is None:
                continue

            score_val = sample.score.value
            score_answer = sample.score.answer or ""
            metadata = sample.score.metadata or {}

            rows.append({
                "model": model,
                "condition": condition,
                "score": score_val,
                "category": score_answer,
                "risk_domain": metadata.get("risk_domain", "unknown"),
                "risk_subdomain": metadata.get("risk_subdomain", "unknown"),
                "parse_error": metadata.get("parse_error", False),
                "sample_id": sample.id or "",
            })

    return pd.DataFrame(rows)


def analyze(df: pd.DataFrame):
    """Run all analyses and print results."""
    if df.empty:
        print("No data to analyze.")
        return

    # 1. Sample counts per condition x model
    print("=" * 70)
    print("SAMPLE COUNTS BY CONDITION x MODEL")
    print("=" * 70)
    counts = df.pivot_table(
        values="score",
        index="model",
        columns="condition",
        aggfunc="count",
    )
    col_order = [
        c
        for c in [
            "bare_chat",
            "system_prompt_only",
            "tools_only",
            "system_prompt_and_tools",
        ]
        if c in counts.columns
    ]
    counts = counts[col_order]
    print(tabulate(counts, headers="keys", tablefmt="grid", floatfmt=".0f"))
    print()

    # 2. Mean refusal score by condition x model
    print("=" * 70)
    print("MEAN REFUSAL SCORE BY CONDITION x MODEL")
    print("(1.0 = full refusal, 0.0 = full compliance)")
    print("=" * 70)

    pivot = df.pivot_table(
        values="score",
        index="model",
        columns="condition",
        aggfunc="mean",
    )
    pivot = pivot[col_order]
    print(tabulate(pivot, headers="keys", tablefmt="grid", floatfmt=".3f"))
    print()

    # 3. Category distribution by condition
    print("=" * 70)
    print("CATEGORY DISTRIBUTION BY CONDITION")
    print("=" * 70)

    cat_dist = (
        df.groupby(["condition", "category"])
        .size()
        .unstack(fill_value=0)
    )
    cat_pct = cat_dist.div(cat_dist.sum(axis=1), axis=0) * 100
    print(tabulate(cat_pct, headers="keys", tablefmt="grid", floatfmt=".1f"))
    print()

    # 4. Subgroup analysis by risk_domain
    print("=" * 70)
    print("MEAN REFUSAL SCORE BY RISK DOMAIN x CONDITION")
    print("=" * 70)

    domain_pivot = df.pivot_table(
        values="score",
        index="risk_domain",
        columns="condition",
        aggfunc="mean",
    )
    if col_order:
        domain_pivot = domain_pivot[[c for c in col_order if c in domain_pivot.columns]]
    print(tabulate(domain_pivot, headers="keys", tablefmt="grid", floatfmt=".3f"))
    print()

    # 5. Parse error rate
    n_errors = df["parse_error"].sum()
    n_total = len(df)
    print(f"Parse errors: {n_errors}/{n_total} ({n_errors / n_total * 100:.1f}%)")

    # 6. Sample counts
    print(f"\nTotal samples scored: {n_total}")
    print(f"Models: {df['model'].nunique()}")
    print(f"Conditions: {df['condition'].nunique()}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/01_unrelated_tools",
        help="Directory containing eval logs",
    )
    args = parser.parse_args()

    df = load_logs(args.log_dir)
    analyze(df)


if __name__ == "__main__":
    main()
