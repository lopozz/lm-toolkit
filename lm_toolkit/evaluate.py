from lm_toolkit.benchmarks.tool_call import evaluate_tool_call


def evaluate(
    model: str,
    tasks: list[dict],
    backend,
    benchmark: str,
    kwargs: dict | None = None,
):
    kwargs = kwargs or {}

    if benchmark == "tool_call":
        return evaluate_tool_call(
            model=model,
            tasks=tasks,
            backend=backend,
            **kwargs,
        )

    raise ValueError(f"Unsupported benchmark: {benchmark}")
