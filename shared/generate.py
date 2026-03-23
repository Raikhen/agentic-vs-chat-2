"""Custom generate solver that converts API-level safety refusals into scoreable outputs."""

from inspect_ai.model import ModelOutput
from inspect_ai.solver import Generate, TaskState, generate, solver

# Error patterns indicating the provider refused on safety grounds
_SAFETY_REFUSAL_PATTERNS = [
    "limited access to this content for safety reasons",
    "content management policy",
    "content policy violation",
]

_API_REFUSAL_OUTPUT = (
    "I'm unable to help with this request. "
    "This content has been flagged by the model provider's safety system."
)


def _matches_any(error_msg: str, patterns: list[str]) -> bool:
    msg_lower = error_msg.lower()
    return any(p.lower() in msg_lower for p in patterns)


@solver
def resilient_generate():
    """Wrap generate() to convert API safety refusals into scoreable outputs.

    When a model provider returns a safety refusal error (HTTP 400 with content
    policy message), this solver catches it and produces a ModelOutput with
    stop_reason="content_filter" so the scorer can grade it as a refusal.

    All other errors (context overflow, rate limits, etc.) propagate normally.
    """
    inner = generate()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        try:
            return await inner(state, generate)
        except Exception as e:
            error_msg = str(e)
            if _matches_any(error_msg, _SAFETY_REFUSAL_PATTERNS):
                state.output = ModelOutput.from_content(
                    model="api_safety_refusal",
                    content=_API_REFUSAL_OUTPUT,
                    stop_reason="content_filter",
                )
                state.metadata["api_safety_refusal"] = True
                return state
            raise

    return solve
