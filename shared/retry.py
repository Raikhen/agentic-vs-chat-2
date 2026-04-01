"""Shared retry infrastructure for re-running missing samples.

Scans all .eval files in a log directory, determines which (model, condition,
sample_id) triples completed successfully, and re-runs only the missing ones.
"""

import json
import os
import zipfile
from collections import defaultdict
from typing import Callable

from inspect_ai import Task, eval

from shared.dataset import fortress_dataset
from shared.models import DEFAULT_MODELS, canonical_model


def scan_completed_samples(
    log_dir: str,
    conditions: list[str],
    condition_from_log: Callable[[str, zipfile.ZipFile, dict], str | None] | None = None,
) -> dict[tuple[str, str], set[int]]:
    """Return {(canonical_model, condition): {completed_sample_ids}}.

    Args:
        log_dir: Directory containing .eval files.
        conditions: Valid condition names (used to filter).
        condition_from_log: Optional callable ``(filename, zipfile, start_json) -> condition``.
            If None, reads condition from ``start_json["eval"]["task"]``.
    """
    completed: dict[tuple[str, str], set[int]] = defaultdict(set)

    for filename in os.listdir(log_dir):
        if not filename.endswith(".eval"):
            continue

        filepath = os.path.join(log_dir, filename)

        try:
            z = zipfile.ZipFile(filepath)
            start = json.loads(z.read("_journal/start.json"))
            model = canonical_model(start.get("eval", {}).get("model", ""))

            if condition_from_log is not None:
                condition = condition_from_log(filename, z, start)
            else:
                condition = start.get("eval", {}).get("task", "")

            if condition not in conditions:
                z.close()
                continue

            for name in z.namelist():
                if not name.startswith("samples/"):
                    continue
                data = json.loads(z.read(name))
                if not data.get("error") and data.get("id") is not None:
                    completed[(model, condition)].add(data["id"])
            z.close()
        except Exception as e:
            print(f"  Warning: could not read {filename}: {e}")

    return completed


def run_retry_loop(
    args,
    conditions: list[str],
    make_task: Callable[[str], Task],
    condition_from_log: Callable[[str, zipfile.ZipFile, dict], str | None] | None = None,
    models: list[str] | None = None,
):
    """Scan completed samples, find missing ones, re-run them.

    Args:
        args: Parsed argparse namespace with ``log_dir``, ``dry_run``,
            ``max_connections`` attributes.
        conditions: List of condition names to retry.
        make_task: Experiment-specific function ``(condition) -> Task``.
        condition_from_log: Passed to :func:`scan_completed_samples`.
        models: Model list to retry for.  Defaults to :data:`DEFAULT_MODELS`.
    """
    if models is None:
        models = DEFAULT_MODELS

    print("Scanning previous eval logs...")
    completed = scan_completed_samples(args.log_dir, conditions, condition_from_log)

    # Get all sample IDs (inspect_ai auto-assigns 1-based sequential IDs)
    full_ds = fortress_dataset()
    total_samples = len(full_ds)
    all_ids = list(range(1, total_samples + 1))
    print(f"Dataset has {total_samples} samples\n")

    # Determine what's missing for each (model, condition)
    max_cond_len = max(len(c) for c in conditions)
    cond_width = max(max_cond_len + 2, 15)

    runs = []  # list of (condition, model, missing_ids)
    total_missing = 0
    total_done = 0

    print(f"{'Model':<45} {'Condition':<{cond_width}} {'Done':>5} {'Missing':>7}")
    print("-" * (45 + cond_width + 14))

    for model in models:
        for condition in conditions:
            done_ids = completed.get((model, condition), set())
            missing_ids = [sid for sid in all_ids if sid not in done_ids]

            n_done = len(done_ids)
            n_missing = len(missing_ids)
            total_done += n_done
            total_missing += n_missing

            status = "  SKIP" if n_missing == 0 else ""
            print(
                f"{model:<45} {condition:<{cond_width}} {n_done:>5} {n_missing:>7}{status}"
            )

            if missing_ids:
                runs.append((condition, model, missing_ids))

    print("-" * (45 + cond_width + 14))
    print(f"{'TOTAL':<45} {'':<{cond_width}} {total_done:>5} {total_missing:>7}")
    print(f"\n{len(runs)} evals to run, {total_missing} total samples")

    if args.dry_run:
        print("\n--dry-run specified, exiting.")
        return

    if not runs:
        print("\nAll samples already completed!")
        return

    # Run each (condition, model) pair with only its missing sample IDs.
    # Sequential per pair, but inspect_ai parallelises samples within each.
    print(f"\nStarting retry run ({len(runs)} evals)...\n")

    for i, (condition, model, missing_ids) in enumerate(runs):
        n = len(missing_ids)
        print(f"[{i+1}/{len(runs)}] {condition} x {model}: {n} samples")

        task = make_task(condition)
        results = eval(
            tasks=task,
            model=model,
            sample_id=missing_ids,
            max_tasks=1,
            max_connections=args.max_connections,
            fail_on_error=False,
            log_dir=args.log_dir,
        )

        for log in results:
            if log.results and log.results.scores:
                for score_log in log.results.scores:
                    mean_val = score_log.metrics.get("mean", None)
                    print(f"    mean={mean_val.value:.3f}" if mean_val else "    N/A")
            else:
                print("    NO RESULTS")

    print("\n" + "=" * 70)
    print("RETRY COMPLETE")
    print("New .eval files written to:", args.log_dir)
    print("=" * 70)
