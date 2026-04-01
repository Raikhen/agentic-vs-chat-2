"""G0DM0D3 API client for plan generation and other integrations."""

import os

import requests

DEFAULT_URL = "http://localhost:7860"
DEFAULT_MODEL = "deepseek/deepseek-v3.2"


class GodmodeClient:
    """Synchronous client for the G0DM0D3 REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        openrouter_api_key: str | None = None,
    ):
        self.base_url = (
            base_url or os.environ.get("GODMODE_API_URL", DEFAULT_URL)
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("GODMODE_API_KEY", "")
        self.openrouter_api_key = openrouter_api_key or os.environ.get(
            "OPENROUTER_API_KEY", ""
        )

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def chat_completion(
        self,
        messages: list[dict],
        model: str = DEFAULT_MODEL,
        godmode: bool = True,
        autotune: bool = False,
        parseltongue: bool = False,
        stm_modules: list[str] | None = None,
        temperature: float = 0,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> str:
        """Send a chat completion request and return the response text.

        Uses the OpenAI-compatible /v1/chat/completions endpoint, which
        transparently applies the G0DM0D3 pipeline (GODMODE system prompt,
        AutoTune, Parseltongue, STM) behind the standard interface.
        """
        body: dict = {
            "model": model,
            "messages": messages,
            "godmode": godmode,
            "autotune": autotune,
            "parseltongue": parseltongue,
            "stm_modules": stm_modules or [],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "contribute_to_dataset": False,
        }
        if self.openrouter_api_key:
            body["openrouter_api_key"] = self.openrouter_api_key

        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self._headers(),
            json=body,
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()

        # OpenAI-compatible response format
        return data["choices"][0]["message"]["content"]
