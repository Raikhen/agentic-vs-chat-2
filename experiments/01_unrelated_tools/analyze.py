"""Post-run analysis: aggregate results and subgroup breakdowns.

Handles multiple .eval files per (model, condition) by deduplicating samples.
Normalizes model names across different routing providers.
"""

import argparse

import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log
from tabulate import tabulate

# Map old routing names to canonical names used in the current run config
_CANONICAL_MODEL = {
    "openrouter/anthropic/claude-haiku-4.5": "anthropic_fallback/claude-haiku-4.5",
    "openrouter/openai/gpt-5.4-mini": "openai/gpt-5.4-mini",
}


def _canonical_model(model: str) -> str:
    return _CANONICAL_MODEL.get(model, model)


def load_logs(log_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all eval logs, deduplicate by (model, condition, sample_id).

    When a sample appears in multiple log files, the successfully scored
    version is kept.  Among multiple scored versions the latest is kept.
    """
    log_refs = list_eval_logs(log_dir)
    scored: dict[tuple, dict] = {}   # (model, condition, sample_id) -> row
    errors: dict[tuple, dict] = {}   # same key -> error row (only if no scored version)

    for ref in log_refs:
        log = read_eval_log(ref.name)
        model = _canonical_model(log.eval.model)
        condition = log.eval.task

        if not log.samples:
            continue

        for sample in log.samples:
            sid = sample.id or ""
            key = (model, condition, sid)

            if sample.score is None:
                # Only keep the error if we don't already have a scored version
                if key not in scored and key not in errors:
                    errors[key] = {
                        "model": model,
                        "condition": condition,
                        "sample_id": sid,
                        "error": getattr(sample, "error", None) or "unknown",
                    }
                continue

            score_val = sample.score.value
            score_answer = sample.score.answer or ""
            metadata = sample.score.metadata or {}

            scored[key] = {
                "model": model,
                "condition": condition,
                "score": score_val,
                "category": score_answer,
                "risk_domain": metadata.get("risk_domain", "unknown"),
                "risk_subdomain": metadata.get("risk_subdomain", "unknown"),
                "parse_error": metadata.get("parse_error", False),
                "api_safety_refusal": metadata.get("api_safety_refusal", False),
                "sample_id": sid,
            }
            # Remove from errors if we now have a scored version
            errors.pop(key, None)

    return pd.DataFrame(list(scored.values())), pd.DataFrame(list(errors.values()))


def analyze(df: pd.DataFrame, errors_df: pd.DataFrame | None = None):
    """Run all analyses and print results."""
    if df.empty:
        print("No data to analyze.")
        return

    # 0. Report errored/skipped samples
    if errors_df is not None and not errors_df.empty:
        print("=" * 70)
        print("ERRORED SAMPLES (excluded from analysis)")
        print("=" * 70)
        error_counts = errors_df.groupby(["model", "condition"]).size()
        print(tabulate(
            error_counts.reset_index(name="count"),
            headers="keys",
            tablefmt="grid",
            showindex=False,
        ))
        print(f"\nTotal errored samples: {len(errors_df)}")
        print()

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

    # 6. API safety refusals (caught by resilient_generate, scored as refusals)
    n_api_refusals = df["api_safety_refusal"].sum()
    print(f"API safety refusals (scored as content_filter): {n_api_refusals}/{n_total} ({n_api_refusals / n_total * 100:.1f}%)")

    # 7. Sample counts
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

    df, errors_df = load_logs(args.log_dir)
    analyze(df, errors_df)


if __name__ == "__main__":
    main()
