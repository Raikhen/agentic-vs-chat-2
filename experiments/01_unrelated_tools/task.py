"""Task definitions for the 4 experimental conditions."""

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import system_message, use_tools
from inspect_ai.tool import python, text_editor

import shared.persistent_sandbox  # noqa: F401  # register persistent_docker sandbox

from shared.dataset import fortress_dataset
from shared.generate import resilient_generate
from shared.scorer import refusal_scorer
from shared.system_prompts import CODING_ASSISTANT_PROMPT


def _make_task(
    condition: str,
    include_system_prompt: bool = False,
    include_tools: bool = False,
    limit: int | None = None,
) -> Task:
    """Factory for creating experimental condition tasks."""
    solvers = []

    if include_system_prompt:
        solvers.append(system_message(CODING_ASSISTANT_PROMPT))

    if include_tools:
        solvers.append(use_tools([python(timeout=30), text_editor()]))

    solvers.append(resilient_generate())

    sandbox = "persistent_docker" if include_tools else None

    return Task(
        dataset=fortress_dataset(limit=limit),
        solver=solvers,
        scorer=refusal_scorer(),
        config=GenerateConfig(temperature=0, max_tokens=4096),
        sandbox=sandbox,
        metadata={"condition": condition},
    )


@task
def bare_chat(limit: int | None = None) -> Task:
    """Baseline: no system prompt, no tools."""
    return _make_task("bare_chat", limit=limit)


@task
def system_prompt_only(limit: int | None = None) -> Task:
    """System prompt only, no tools."""
    return _make_task(
        "system_prompt_only",
        include_system_prompt=True,
        limit=limit,
    )


@task
def tools_only(limit: int | None = None) -> Task:
    """Tools only (with Docker sandbox), no system prompt."""
    return _make_task(
        "tools_only",
        include_tools=True,
        limit=limit,
    )


@task
def system_prompt_and_tools(limit: int | None = None) -> Task:
    """Both system prompt and tools (with Docker sandbox)."""
    return _make_task(
        "system_prompt_and_tools",
        include_system_prompt=True,
        include_tools=True,
        limit=limit,
    )
