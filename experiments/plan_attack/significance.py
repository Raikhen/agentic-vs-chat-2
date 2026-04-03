"""Paired Wilcoxon signed-rank tests with Bonferroni correction for plan_attack."""

import numpy as np
from scipy import stats
from shared.analysis import load_logs


def main():
    df, errors_df = load_logs("logs/plan_attack")

    models = sorted(df["model"].unique())
    n_tests = len(models)
    alpha = 0.05
    bonferroni_alpha = alpha / n_tests

    print(f"Models: {n_tests}, Bonferroni-adjusted alpha: {bonferroni_alpha:.4f}")
    print(
        f"{'Model':<30} {'n_paired':>8} {'raw_mean':>9} {'plan_mean':>10} "
        f"{'delta':>7} {'p-value':>10} {'sig?':>5}"
    )
    print("-" * 85)

    for model in models:
        mdf = df[df["model"] == model]
        raw_df = mdf[mdf["condition"] == "raw"].set_index("sample_id")["score"]
        plan_df = mdf[mdf["condition"] == "plan"].set_index("sample_id")["score"]

        # Match by sample_id
        common_ids = sorted(set(raw_df.index) & set(plan_df.index))
        if len(common_ids) < 10:
            print(f"{model:<30} {len(common_ids):>8} -- too few paired samples")
            continue

        raw_vals = np.array([raw_df[sid] for sid in common_ids])
        plan_vals = np.array([plan_df[sid] for sid in common_ids])

        diffs = plan_vals - raw_vals

        # Wilcoxon signed-rank test (two-sided)
        nonzero_diffs = diffs[diffs != 0]
        if len(nonzero_diffs) < 10:
            p_val = 1.0
        else:
            _stat, p_val = stats.wilcoxon(nonzero_diffs)

        sig = "*" if p_val < bonferroni_alpha else ""

        raw_mean = np.mean(raw_vals)
        plan_mean = np.mean(plan_vals)
        delta = plan_mean - raw_mean

        # Bootstrap 95% CI on delta
        rng = np.random.default_rng(42)
        boot_deltas = []
        for _ in range(10000):
            idx = rng.integers(0, len(diffs), size=len(diffs))
            boot_deltas.append(np.mean(diffs[idx]))
        ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])

        print(
            f"{model:<30} {len(common_ids):>8} {raw_mean:>9.3f} {plan_mean:>10.3f} "
            f"{delta:>+7.3f} {p_val:>10.2e} {sig:>5}"
        )
        print(
            f"  {'':30} Bootstrap 95% CI on delta: [{ci_lo:+.3f}, {ci_hi:+.3f}]"
        )

    print()
    print("Note: positive delta = plan increases refusal (model resists more)")
    print("      negative delta = plan decreases refusal (model complies more)")


if __name__ == "__main__":
    main()
