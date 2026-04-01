"""Task definitions for the plan-attack experiment (raw vs plan conditions)."""

import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig

from shared.dataset import fortress_dataset
from shared.generate import resilient_generate
from shared.scorer import refusal_scorer

PLANS_FILE = Path("data/plan_attack/plans.jsonl")


def _load_plan_dataset(limit: int | None = None) -> list[Sample]:
    """Load plan versions of adversarial prompts from the pre-generated JSONL file."""
    if not PLANS_FILE.exists():
        raise FileNotFoundError(
            f"Plans file not found: {PLANS_FILE}\n"
            "Run 'python -m experiments.plan_attack.generate_plans' first."
        )

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
                        "original_prompt": entry.get("adversarial_prompt", ""),
                    },
                )
            )

    # Sort by sample_id for consistency
    samples.sort(key=lambda s: s.id)

    if limit is not None:
        samples = samples[:limit]

    return samples


def _make_task(condition: str, dataset) -> Task:
    """Factory for creating experimental condition tasks."""
    return Task(
        dataset=dataset,
        solver=[resilient_generate()],
        scorer=refusal_scorer(),
        config=GenerateConfig(temperature=0, max_tokens=4096),
        message_limit=10,
        metadata={"condition": condition},
    )


@task
def raw(limit: int | None = None) -> Task:
    """Baseline: original adversarial prompts (bare chat)."""
    return _make_task("raw", fortress_dataset(limit=limit))


@task
def plan(limit: int | None = None) -> Task:
    """Plan condition: structured plan versions of adversarial prompts."""
    return _make_task("plan", _load_plan_dataset(limit=limit))
