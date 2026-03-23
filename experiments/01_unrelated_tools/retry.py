"""Retry only missing samples, preserving data from all previous runs.

Scans all .eval files in the log directory, determines which (model, condition,
sample_id) triples completed successfully, and re-runs only the missing ones.

Usage:
    python -m experiments.01_unrelated_tools.retry [--dry-run] [--log-dir DIR]
"""

import argparse
import json
import os
import zipfile
from collections import defaultdict

import shared.fallback_model  # noqa: F401
import shared.persistent_sandbox  # noqa: F401

from inspect_ai import Task, eval
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import system_message, use_tools
from inspect_ai.tool import python, text_editor

from shared.dataset import fortress_dataset
from shared.generate import resilient_generate
from shared.scorer import refusal_scorer
from shared.system_prompts import CODING_ASSISTANT_PROMPT

DEFAULT_MODELS = [
    "anthropic_fallback/claude-haiku-4.5",
    "openai/gpt-5.4-mini",
    "openrouter/google/gemini-3-flash-preview",
    "openrouter/deepseek/deepseek-v3.2",
    "openrouter/qwen/qwen3.5-flash-02-23",
    "openrouter/x-ai/grok-4-fast",
]

CONDITIONS = {
    "bare_chat": {"system_prompt": False, "tools": False},
    "system_prompt_only": {"system_prompt": True, "tools": False},
    "tools_only": {"system_prompt": False, "tools": True},
    "system_prompt_and_tools": {"system_prompt": True, "tools": True},
}

CONDITION_FROM_FILENAME = {
    "bare-chat": "bare_chat",
    "system-prompt-only": "system_prompt_only",
    "tools-only": "tools_only",
    "system-prompt-and-tools": "system_prompt_and_tools",
}

# Map old routing names to current DEFAULT_MODELS names
CANONICAL_MODEL = {
    "openrouter/anthropic/claude-haiku-4.5": "anthropic_fallback/claude-haiku-4.5",
    "openrouter/openai/gpt-5.4-mini": "openai/gpt-5.4-mini",
}


def canonical_model(model: str) -> str:
    return CANONICAL_MODEL.get(model, model)


def condition_from_filename(filename: str) -> str | None:
    for pattern, condition in CONDITION_FROM_FILENAME.items():
        if pattern in filename:
            return condition
    return None


def scan_completed_samples(log_dir: str) -> dict[tuple[str, str], set[int]]:
    """Return {(canonical_model, condition): {completed_sample_ids}}."""
    completed: dict[tuple[str, str], set[int]] = defaultdict(set)

    for filename in os.listdir(log_dir):
        if not filename.endswith(".eval"):
            continue

        filepath = os.path.join(log_dir, filename)
        condition = condition_from_filename(filename)
        if condition is None:
            continue

        try:
            z = zipfile.ZipFile(filepath)
            start = json.loads(z.read("_journal/start.json"))
            model = canonical_model(start.get("eval", {}).get("model", ""))

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


def make_task(condition: str) -> Task:
    """Build a Task for the given condition with the full dataset."""
    cfg = CONDITIONS[condition]
    solvers = []

    if cfg["system_prompt"]:
        solvers.append(system_message(CODING_ASSISTANT_PROMPT))
    if cfg["tools"]:
        solvers.append(use_tools([python(timeout=30), text_editor()]))
    solvers.append(resilient_generate())

    sandbox = "persistent_docker" if cfg["tools"] else None

    return Task(
        name=condition,
        dataset=fortress_dataset(),
        solver=solvers,
        scorer=refusal_scorer(),
        config=GenerateConfig(temperature=0, max_tokens=4096),
        sandbox=sandbox,
        metadata={"condition": condition},
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retry only missing samples from previous runs"
    )
    parser.add_argument(
        "--log-dir", default="logs/01_unrelated_tools",
        help="Directory containing previous eval logs",
    )
    parser.add_argument(
        "--max-tasks", type=int, default=24,
        help="Max concurrent task×model evaluations",
    )
    parser.add_argument(
        "--max-connections", type=int, default=50,
        help="Max concurrent API connections per model",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only print what would be run",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Scanning previous eval logs...")
    completed = scan_completed_samples(args.log_dir)

    # Get all sample IDs (inspect_ai auto-assigns 1-based sequential IDs)
    full_ds = fortress_dataset()
    total_samples = len(full_ds)
    all_ids = list(range(1, total_samples + 1))
    print(f"Dataset has {total_samples} samples\n")

    # Determine what's missing for each (model, condition)
    runs = []  # list of (condition, model, missing_ids)
    total_missing = 0
    total_done = 0

    print(f"{'Model':<45} {'Condition':<28} {'Done':>5} {'Missing':>7}")
    print("-" * 90)

    for model in DEFAULT_MODELS:
        for condition in CONDITIONS:
            done_ids = completed.get((model, condition), set())
            missing_ids = [sid for sid in all_ids if sid not in done_ids]

            n_done = len(done_ids)
            n_missing = len(missing_ids)
            total_done += n_done
            total_missing += n_missing

            status = "  SKIP" if n_missing == 0 else ""
            print(
                f"{model:<45} {condition:<28} {n_done:>5} {n_missing:>7}{status}"
            )

            if missing_ids:
                runs.append((condition, model, missing_ids))

    print("-" * 90)
    print(f"{'TOTAL':<45} {'':<28} {total_done:>5} {total_missing:>7}")
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
    print("Use the analysis script to merge results from all .eval files.")
    print("=" * 70)


if __name__ == "__main__":
    main()
