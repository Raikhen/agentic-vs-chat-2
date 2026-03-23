"""Anthropic fallback ModelAPI: tries direct Anthropic API first, falls back to OpenRouter."""

import logging
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelAPI,
    ModelCall,
    ModelOutput,
    get_model,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

logger = logging.getLogger(__name__)


def _should_fallback(exc: Exception) -> bool:
    """Return True if the exception warrants falling back to OpenRouter."""
    msg = str(exc).lower()

    # Credit/billing errors
    credit_phrases = [
        "your credit balance is too low",
        "payment required",
        "insufficient credit",
        "insufficient funds",
        "out of credits",
        "no credits",
        "credit balance",
        "billing",
        "exceeded your current quota",
    ]
    if any(phrase in msg for phrase in credit_phrases):
        return True

    status_code = getattr(exc, "status_code", None)

    # 403 often means billing/permissions issue
    if status_code == 403:
        return True

    # 404 means model not found on this provider
    if status_code == 404:
        return True

    return False


@modelapi("anthropic_fallback")
class AnthropicFallbackAPI(ModelAPI):
    """ModelAPI that tries direct Anthropic first, falls back to OpenRouter on credit errors."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
        )

        # Build the two underlying Model instances (which wrap their own ModelAPI).
        # get_model() will resolve the correct provider and API keys from env vars.
        self._primary_model = get_model(
            f"anthropic/{model_name}",
            config=config,
            memoize=False,
        )
        self._fallback_model = get_model(
            f"openrouter/anthropic/{model_name}",
            config=config,
            memoize=False,
        )
        self._using_fallback = False

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput | tuple[ModelOutput | Exception, ModelCall]:
        # If we've already switched to fallback permanently, skip trying primary
        if self._using_fallback:
            return await self._fallback_model.api.generate(
                input, tools, tool_choice, config
            )

        try:
            result = await self._primary_model.api.generate(
                input, tools, tool_choice, config
            )

            # The Anthropic provider returns tuple[ModelOutput | Exception, ModelCall].
            # If the ModelOutput slot is actually an Exception, check if it's a credit error.
            if isinstance(result, tuple):
                output, model_call = result
                if isinstance(output, Exception):
                    if _should_fallback(output):
                        logger.warning(
                            "Anthropic credit error (%s), falling back to OpenRouter for %s",
                            output,
                            self.model_name,
                        )
                        self._using_fallback = True
                        return await self._fallback_model.api.generate(
                            input, tools, tool_choice, config
                        )
                return result
            return result

        except Exception as e:
            if _should_fallback(e):
                logger.warning(
                    "Anthropic credit error (%s), falling back to OpenRouter for %s",
                    e,
                    self.model_name,
                )
                self._using_fallback = True
                return await self._fallback_model.api.generate(
                    input, tools, tool_choice, config
                )
            raise

    async def aclose(self) -> None:
        """Close both underlying model API clients."""
        await self._primary_model.api.aclose()
        await self._fallback_model.api.aclose()
