"""Orchestration: run all 4 conditions across subject models."""

import argparse
import importlib

import shared.fallback_model  # noqa: F401  # register anthropic_fallback provider
import shared.persistent_sandbox  # noqa: F401  # register persistent_docker sandbox

from inspect_ai import eval

from shared.models import DEFAULT_MODELS

_task_mod = importlib.import_module("experiments.unrelated_tools.task")
bare_chat = _task_mod.bare_chat
system_prompt_only = _task_mod.system_prompt_only
tools_only = _task_mod.tools_only
system_prompt_and_tools = _task_mod.system_prompt_and_tools


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the unrelated-tools refusal experiment"
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
        default=24,
        help="Max concurrent task×model evaluations (default: all 24)",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=50,
        help="Max concurrent API connections per model",
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=None,
        help="Abort if estimated cost exceeds this (USD)",
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
        default="logs/unrelated_tools",
        help="Directory for eval logs",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Build tasks for each condition
    tasks = [
        bare_chat(limit=args.limit),
        system_prompt_only(limit=args.limit),
        tools_only(limit=args.limit),
        system_prompt_and_tools(limit=args.limit),
    ]

    # Run all conditions across all models
    results = eval(
        tasks,
        model=args.models,
        max_tasks=args.max_tasks,
        max_connections=args.max_connections,
        fail_on_error=False,
        log_dir=args.log_dir,
    )

    # Print summary
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
                print(f"  {model:<45} {task_name:<30} mean={mean_str}")
        else:
            print(f"  {model:<45} {task_name:<30} NO RESULTS")

    print("=" * 70)


if __name__ == "__main__":
    main()
