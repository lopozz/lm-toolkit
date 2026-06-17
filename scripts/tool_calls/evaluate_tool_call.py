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
"""

import argparse

from openai import OpenAI
from lm_toolkit.benchmarks.tool_call import (
    print_result,
    build_test_cases,
    load_config,
    ask_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether a local vLLM model calls a specified tool."
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible vLLM endpoint.",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key sent to the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name served by vLLM.",
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
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    tool = config["tool"]
    system_prompt = config["system_prompt"]
    test_cases = build_test_cases(config)

    positives_total = 0
    positives_passed = 0
    negatives_total = 0
    negatives_passed = 0
    argument_total = 0
    argument_passed = 0

    for index, test_case in enumerate(test_cases, start=1):
        (
            did_call_expected_tool,
            did_match_arguments,
            called_tool_names,
            actual_arguments,
        ) = ask_model(
            client=client,
            model=args.model,
            system_prompt=system_prompt,
            tool=tool,
            test_case=test_case,
            temperature=args.temperature,
            debug=args.debug,
        )

        call_ok = did_call_expected_tool == test_case.should_call

        if test_case.should_call:
            positives_total += 1
            positives_passed += int(call_ok)

            if test_case.expected_arguments is not None:
                argument_total += 1
                argument_passed += int(did_call_expected_tool and did_match_arguments)
        else:
            negatives_total += 1
            negatives_passed += int(call_ok)

        print_result(
            index=index,
            test_case=test_case,
            did_call_expected_tool=did_call_expected_tool,
            did_match_arguments=did_match_arguments,
            called_tool_names=called_tool_names,
            actual_arguments=actual_arguments,
        )

    total = positives_total + negatives_total
    passed = positives_passed + negatives_passed

    print()
    print("Summary")
    print(f"Config:          {args.config}")
    print(f"Tool:            {tool['function']['name']}")
    print(f"Should call:     {positives_passed}/{positives_total} passed")
    print(f"Should not call: {negatives_passed}/{negatives_total} passed")
    print(f"Total calls:     {passed}/{total} passed")

    if argument_total:
        print(f"Arguments:       {argument_passed}/{argument_total} passed")


if __name__ == "__main__":
    main()
