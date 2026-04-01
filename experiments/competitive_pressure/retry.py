"""Retry only missing samples for the competitive-pressure experiment.

Scans all .eval files in the log directory, determines which (model, condition,
sample_id) triples completed successfully, and re-runs only the missing ones.

Usage:
    python -m experiments.competitive_pressure.retry [--dry-run] [--log-dir DIR]
"""

import argparse

import shared.fallback_model  # noqa: F401

from inspect_ai import Task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import system_message

from shared.dataset import fortress_dataset
from shared.generate import resilient_generate
from shared.retry import run_retry_loop
from shared.scorer import refusal_scorer

from experiments.competitive_pressure.task import (
    COMPETITION_SUFFIX,
    DEPTH_DIRECTIVE_BODY,
    NEUTRAL_SUFFIX,
    NO_SUFFIX,
)

CONDITIONS = {
    "no_suffix": NO_SUFFIX,
    "neutral_suffix": NEUTRAL_SUFFIX,
    "competition": COMPETITION_SUFFIX,
}


def make_task(condition: str) -> Task:
    """Build a Task for the given condition with the full dataset."""
    suffix = CONDITIONS[condition]
    prompt = DEPTH_DIRECTIVE_BODY + suffix

    return Task(
        name=condition,
        dataset=fortress_dataset(),
        solver=[
            system_message(prompt),
            resilient_generate(),
        ],
        scorer=refusal_scorer(),
        config=GenerateConfig(temperature=0, max_tokens=4096),
        message_limit=10,
        metadata={"condition": condition},
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retry missing samples for competitive-pressure experiment"
    )
    parser.add_argument(
        "--log-dir",
        default="logs/competitive_pressure",
        help="Directory containing previous eval logs",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=50,
        help="Max concurrent API connections per model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_retry_loop(
        args,
        conditions=list(CONDITIONS.keys()),
        make_task=make_task,
    )


if __name__ == "__main__":
    main()
