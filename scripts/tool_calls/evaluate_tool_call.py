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

import json
import yaml
import argparse

from typing import Any
from pathlib import Path
from openai import OpenAI
from dataclasses import dataclass


@dataclass(frozen=True)
class TestCase:
    text: str
    should_call: bool
    expected_arguments: dict[str, Any] | None = None


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


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config: {config_path}")

    required_keys = ["tool", "system_prompt", "should_call_tool", "should_not_call_tool"]
    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        raise ValueError(f"Missing required config keys: {', '.join(missing_keys)}")

    return config


def build_test_cases(config: dict[str, Any]) -> list[TestCase]:
    test_cases: list[TestCase] = []

    for item in config["should_call_tool"]:
        if isinstance(item, str):
            test_cases.append(TestCase(text=item, should_call=True))
        else:
            test_cases.append(
                TestCase(
                    text=item["text"],
                    should_call=True,
                    expected_arguments=item.get("expected_arguments"),
                )
            )

    for item in config["should_not_call_tool"]:
        if isinstance(item, str):
            test_cases.append(TestCase(text=item, should_call=False))
        else:
            test_cases.append(TestCase(text=item["text"], should_call=False))

    return test_cases


def parse_tool_arguments(arguments: str | dict[str, Any] | None) -> dict[str, Any]:
    if arguments is None:
        return {}

    if isinstance(arguments, dict):
        return arguments

    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def arguments_match(
    actual_arguments: dict[str, Any],
    expected_arguments: dict[str, Any] | None,
) -> bool:
    """
    Exact match for expected keys only.

    Example:
    expected_arguments = {"nome_studente": "Marco Rossi"}
    actual_arguments   = {"nome_studente": "Marco Rossi"}

    This returns True.

    If the model adds extra keys, they are ignored here because some local
    models may include harmless extra fields.
    """
    if expected_arguments is None:
        return True

    for key, expected_value in expected_arguments.items():
        actual_value = actual_arguments.get(key)

        if actual_value != expected_value:
            return False

    return True


def ask_model(
    client: OpenAI,
    model: str,
    system_prompt: str,
    tool: dict[str, Any],
    test_case: TestCase,
    temperature: float,
    debug: bool,
) -> tuple[bool, bool, list[str], dict[str, Any]]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test_case.text},
        ],
        tools=[tool],
        tool_choice="auto",
        temperature=temperature,
    )

    message = response.choices[0].message
    tool_calls = message.tool_calls or []

    expected_tool_name = tool["function"]["name"]
    called_tool_names = [tool_call.function.name for tool_call in tool_calls]

    matching_tool_call = next(
        (
            tool_call
            for tool_call in tool_calls
            if tool_call.function.name == expected_tool_name
        ),
        None,
    )

    did_call_expected_tool = matching_tool_call is not None

    actual_arguments: dict[str, Any] = {}
    did_match_arguments = True

    if matching_tool_call is not None:
        actual_arguments = parse_tool_arguments(matching_tool_call.function.arguments)
        did_match_arguments = arguments_match(
            actual_arguments=actual_arguments,
            expected_arguments=test_case.expected_arguments,
        )

    if debug:
        print()
        print("DEBUG")
        print("content:", message.content)
        print("tool_calls:", tool_calls)
        print("actual_arguments:", actual_arguments)
        print()

    return (
        did_call_expected_tool,
        did_match_arguments,
        called_tool_names,
        actual_arguments,
    )


def print_result(
    index: int,
    test_case: TestCase,
    did_call_expected_tool: bool,
    did_match_arguments: bool,
    called_tool_names: list[str],
    actual_arguments: dict[str, Any],
) -> None:
    expected = "CALL" if test_case.should_call else "NO CALL"
    actual = "CALL" if did_call_expected_tool else "NO CALL"

    call_ok = did_call_expected_tool == test_case.should_call
    args_ok = did_match_arguments if test_case.should_call else True
    status = "PASS" if call_ok and args_ok else "FAIL"

    called = ", ".join(called_tool_names) if called_tool_names else "-"
    args = json.dumps(actual_arguments, ensure_ascii=False) if actual_arguments else "-"

    print(
        f"{index:02d}. {status} | expected={expected:<7} actual={actual:<7} "
        f"called={called} args={args} | {test_case.text}"
    )


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