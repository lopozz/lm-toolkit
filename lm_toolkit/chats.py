import time

from openai import APIConnectionError, APITimeoutError, OpenAI


def stream_chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    on_delta,
    max_retries: int = 5,
    retry_delay_seconds: float = 5.0,
) -> str:
    """Stream one chat completion, calling on_delta(chunk) as text arrives.

    Returns the full response text. Retries the whole request on transient
    connection errors (e.g. a local server restarting mid-run).
    """
    for attempt in range(1, max_retries + 1):
        text = ""
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    text += delta
                    on_delta(delta)
            return text.strip()
        except (APIConnectionError, APITimeoutError) as error:
            if attempt == max_retries:
                raise
            print(f"[stream_chat_completion] {error!r} (attempt {attempt}/{max_retries}), "
                  f"retrying in {retry_delay_seconds:.0f}s")
            time.sleep(retry_delay_seconds)
