from typing import Any

from openai import OpenAI

from lm_toolkit.backends.base import LMBackend


class OpenAIBackend(LMBackend):
    def __init__(self, base_url: str, api_key: str = "EMPTY"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        return self.client.chat.completions.create(**kwargs)
