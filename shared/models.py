"""Canonical model names and default model list.

Single source of truth — imported by run, analyze, retry, and estimate_cost
modules across all experiments.
"""

# Map old routing names to canonical names used in the current run config
CANONICAL_MODEL = {
    "anthropic_fallback/claude-haiku-4.5": "openrouter/anthropic/claude-haiku-4.5",
    "openrouter/openai/gpt-5.4-mini": "openai/gpt-5.4-mini",
}

DEFAULT_MODELS = [
    "openrouter/anthropic/claude-haiku-4.5",
    "openai/gpt-5.4-mini",
    "openrouter/google/gemini-3-flash-preview",
    "openrouter/deepseek/deepseek-v3.2",
    "openrouter/qwen/qwen3.5-flash-02-23",
    "openrouter/x-ai/grok-4-fast",
]


def canonical_model(model: str) -> str:
    return CANONICAL_MODEL.get(model, model)
