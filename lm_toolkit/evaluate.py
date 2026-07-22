from typing import Any

from lm_toolkit.benchmarks.sparse_retrieval import evaluate_sparse_retrieval
from lm_toolkit.benchmarks.tool_call import evaluate_tool_call


def evaluate(
    model: str,
    tasks: list[dict],
    backend: Any,
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

    if benchmark == "sparse_retrieval":
        return evaluate_sparse_retrieval(
            model=model,
            tasks=tasks,
            backend=backend,
            **kwargs,
        )

    raise ValueError(f"Unsupported benchmark: {benchmark}")
