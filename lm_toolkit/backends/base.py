from typing import Any, Protocol


class LMBackend(Protocol):
    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> Any:
        """Create a chat completion using a backend-specific client."""
        ...
