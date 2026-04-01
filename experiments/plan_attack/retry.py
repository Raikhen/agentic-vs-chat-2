"""Retry only missing samples for the plan-attack experiment.

Scans all .eval files in the log directory, determines which (model, condition,
sample_id) triples completed successfully, and re-runs only the missing ones.

Usage:
    python -m experiments.plan_attack.retry [--dry-run] [--log-dir DIR]
"""

import argparse
import json
from pathlib import Path

import shared.fallback_model  # noqa: F401

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig

from shared.dataset import fortress_dataset
from shared.generate import resilient_generate
from shared.retry import run_retry_loop
from shared.scorer import refusal_scorer

PLANS_FILE = Path("data/plan_attack/plans.jsonl")

CONDITIONS = ["raw", "plan"]


def load_plan_dataset() -> list[Sample]:
    """Load plan dataset from JSONL."""
    samples = []
    with open(PLANS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            samples.append(
                Sample(
                    input=entry["plan_prompt"],
                    target=entry.get("adversarial_prompt", ""),
                    id=entry["sample_id"],
                    metadata={
                        "risk_domain": entry.get("risk_domain", "unknown"),
                        "risk_subdomain": entry.get("risk_subdomain", "unknown"),
                    },
                )
            )
    samples.sort(key=lambda s: s.id)
    return samples


def make_task(condition: str) -> Task:
    """Build a Task for the given condition with the full dataset."""
    if condition == "raw":
        dataset = fortress_dataset()
    else:
        dataset = load_plan_dataset()

    return Task(
        name=condition,
        dataset=dataset,
        solver=[resilient_generate()],
        scorer=refusal_scorer(),
        config=GenerateConfig(temperature=0, max_tokens=4096),
        message_limit=10,
        metadata={"condition": condition},
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retry missing samples for plan-attack experiment"
    )
    parser.add_argument(
        "--log-dir",
        default="logs/plan_attack",
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
        conditions=CONDITIONS,
        make_task=make_task,
    )


if __name__ == "__main__":
    main()
