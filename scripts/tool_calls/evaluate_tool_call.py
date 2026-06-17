"""
Evaluate whether an OpenAI-compatible chat model uses tools correctly.

The script loads a YAML config describing one tool-calling scenario: the tool
schema, the system prompt, examples where the tool should be called, and hard
negative examples where the model should answer without tools.

For each test case, it sends a chat completion request with `tool_choice="auto"`
and checks whether the expected function appears in `message.tool_calls`.
Positive cases may also define expected arguments, allowing the script to verify
both tool selection and argument extraction.

The goal is to measure whether a model can distinguish between requests that
require an external action or retrieval step and requests that should be handled
directly in natural language.

Typical usage:

    python3 scripts/tool_calls/evaluate_tool_call.py \\
      --model mistralai/Ministral-3-3B-Instruct-2512 \\
      --config configs/tools/student_courses.yaml

Use `--debug` to print the raw assistant content, structured tool calls, and
parsed arguments for each request.

Important:
    When evaluating tool calling through vLLM, the server launch command must
    include the parameters required for structured tool-call generation and
    parsing. Passing `tools=[...]` and `tool_choice="auto"` from the client is
    not enough if the vLLM server was not started with tool-calling support.

    For Mistral/Ministral models, include at least:

        --enable-auto-tool-choice
        --tool-call-parser mistral

    See https://docs.vllm.ai/en/stable/features/tool_calling/ for more info
    on funciton calling with vLLM.
    See https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md for more info
    on funciton calling with LlamaCpp.
"""

import argparse

import lm_toolkit

from lm_toolkit.backends import OpenAIBackend
from lm_toolkit.benchmarks.tool_call import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether an OpenAI-compatible model calls a specified tool."
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key sent to the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name served by the backend.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config, e.g. configs/tools/student_courses.yaml",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw model content and tool calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    backend = OpenAIBackend(base_url=args.base_url, api_key=args.api_key)

    lm_toolkit.evaluate(
        model=args.model,
        tasks=[config],
        backend=backend,
        benchmark="tool_call",
        kwargs={
            "temperature": args.temperature,
            "debug": args.debug,
        },
    )


if __name__ == "__main__":
    main()
