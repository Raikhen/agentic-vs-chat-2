"""Orchestration: run raw vs plan conditions across subject models."""

import argparse
import importlib

import shared.fallback_model  # noqa: F401  # register anthropic_fallback provider

from inspect_ai import eval

from shared.models import DEFAULT_MODELS

_task_mod = importlib.import_module("experiments.plan_attack.task")
raw = _task_mod.raw
plan = _task_mod.plan


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the plan-attack refusal experiment"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Subject models to evaluate (OpenRouter model IDs)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=12,
        help="Max concurrent task x model evaluations (default: 12)",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=50,
        help="Max concurrent API connections per model",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per task (for dry runs)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/plan_attack",
        help="Directory for eval logs",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tasks = [
        raw(limit=args.limit),
        plan(limit=args.limit),
    ]

    results = eval(
        tasks,
        model=args.models,
        max_tasks=args.max_tasks,
        max_connections=args.max_connections,
        fail_on_error=False,
        log_dir=args.log_dir,
    )

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for log in results:
        model = log.eval.model
        task_name = log.eval.task
        if log.results and log.results.scores:
            for score_log in log.results.scores:
                metrics = score_log.metrics
                mean_val = metrics.get("mean", None)
                mean_str = f"{mean_val.value:.3f}" if mean_val else "N/A"
                print(f"  {model:<45} {task_name:<15} mean={mean_str}")
        else:
            print(f"  {model:<45} {task_name:<15} NO RESULTS")

    print("=" * 70)


if __name__ == "__main__":
    main()
