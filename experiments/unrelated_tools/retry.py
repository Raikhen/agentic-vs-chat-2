"""Retry only missing samples, preserving data from all previous runs.

Scans all .eval files in the log directory, determines which (model, condition,
sample_id) triples completed successfully, and re-runs only the missing ones.

Usage:
    python -m experiments.unrelated_tools.retry [--dry-run] [--log-dir DIR]
"""

import argparse

import shared.fallback_model  # noqa: F401
import shared.persistent_sandbox  # noqa: F401

from inspect_ai import Task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import system_message, use_tools
from inspect_ai.tool import python, text_editor

from shared.dataset import fortress_dataset
from shared.generate import resilient_generate
from shared.retry import run_retry_loop
from shared.scorer import refusal_scorer
from shared.system_prompts import CODING_ASSISTANT_PROMPT

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


def _condition_from_log(filename, _zf, _start_json):
    """Detect condition from filename (legacy log format)."""
    for pattern, condition in CONDITION_FROM_FILENAME.items():
        if pattern in filename:
            return condition
    return None


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
        message_limit=10,
        sandbox=sandbox,
        metadata={"condition": condition},
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retry only missing samples from previous runs"
    )
    parser.add_argument(
        "--log-dir", default="logs/unrelated_tools",
        help="Directory containing previous eval logs",
    )
    parser.add_argument(
        "--max-tasks", type=int, default=24,
        help="Max concurrent task x model evaluations",
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
    run_retry_loop(
        args,
        conditions=list(CONDITIONS.keys()),
        make_task=make_task,
        condition_from_log=_condition_from_log,
    )


if __name__ == "__main__":
    main()
