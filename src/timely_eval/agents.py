"""OpenAI-compatible asynchronous chat agent wrapper."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

ChatMessage = dict[str, Any]


@dataclass(slots=True)
class ChatModelConfig:
    """Configuration for an OpenAI-compatible chat completion endpoint."""

    model: str
    api_base: str = "https://api.openai.com/v1"
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float | None = 0.7
    max_tokens: int = 4096
    reasoning_effort: str | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)

    def resolved_api_key(self) -> str:
        key = self.api_key or os.getenv(self.api_key_env)
        if not key:
            raise ValueError(
                f"Missing API key. Set --api-key or environment variable {self.api_key_env}."
            )
        return key


class AsyncChatAgent:
    """Small retrying wrapper around ``AsyncOpenAI.chat.completions``."""

    def __init__(
        self,
        config: ChatModelConfig,
        *,
        system_prompt: str,
        role: str = "assistant",
        description: str = "",
        retries: int = 3,
        retry_sleep: float = 0.5,
    ) -> None:
        self.config = config
        self.system_prompt = system_prompt
        self.role = role
        self.description = description
        self.retries = retries
        self.retry_sleep = retry_sleep
        self.client = AsyncOpenAI(
            base_url=config.api_base,
            api_key=config.resolved_api_key(),
        )

    @property
    def model(self) -> str:
        return self.config.model

    async def generate_response(
        self,
        messages: list[ChatMessage],
        *,
        sem: asyncio.Semaphore | None = None,
        **override_params: Any,
    ) -> str:
        """Generate one chat completion response."""

        call_params = self._call_params(override_params)

        async def _call() -> str:
            response = await self.client.chat.completions.create(
                messages=messages,
                **call_params,
            )
            content = response.choices[0].message.content
            return content or ""

        for attempt in range(1, self.retries + 1):
            try:
                if sem is None:
                    return await _call()
                async with sem:
                    return await _call()
            except Exception as exc:  # noqa: BLE001 - keep endpoint errors retryable.
                logger.warning(
                    "Chat completion failed on attempt %s/%s: %s",
                    attempt,
                    self.retries,
                    exc,
                )
                if attempt == self.retries:
                    return f"Error: failed to generate response after {self.retries} attempts."
                await asyncio.sleep(self.retry_sleep)

        return "Error: failed to generate response."

    def _call_params(self, override_params: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.config.model,
            **self.config.extra_params,
            **override_params,
        }

        # Some reasoning models reject temperature and use max_completion_tokens.
        if self.config.model.startswith("gpt-5") or self.config.reasoning_effort:
            params["max_completion_tokens"] = self.config.max_tokens
            if self.config.reasoning_effort:
                params["reasoning_effort"] = self.config.reasoning_effort
        else:
            params["max_tokens"] = self.config.max_tokens
            if self.config.temperature is not None:
                params["temperature"] = self.config.temperature

        return params
