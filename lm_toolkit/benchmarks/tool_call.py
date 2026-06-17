"""
Evaluate whether an OpenAI-compatible chat model uses tools correctly.

The evaluator loads a YAML config describing one tool-calling scenario: the tool
schema, the system prompt, examples where the tool should be called, and hard
negative examples where the model should answer without tools.

For each test case, it sends a chat completion request with `tool_choice="auto"`
and checks whether the expected function appears in `message.tool_calls`.
Positive cases may also define expected arguments, allowing verification of both
tool selection and argument extraction.
"""

import yaml
import json

from typing import Any
from pathlib import Path
from dataclasses import dataclass

from lm_toolkit.backends import LMBackend


@dataclass(frozen=True)
class TestCase:
    text: str
    should_call: bool
    expected_arguments: dict[str, Any] | None = None


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config: {config_path}")

    required_keys = [
        "tool",
        "system_prompt",
        "should_call_tool",
        "should_not_call_tool",
    ]
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


def evaluate_tool_call(
    model: str,
    tasks: list[dict[str, Any]],
    backend: LMBackend,
    temperature: float = 0.0,
    debug: bool = False,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for task in tasks:
        tool = task["tool"]
        system_prompt = task["system_prompt"]
        test_cases = build_test_cases(task)
        task_name = task.get("name", tool["function"]["name"])

        positives_total = 0
        positives_passed = 0
        negatives_total = 0
        negatives_passed = 0
        argument_total = 0
        argument_passed = 0

        for index, test_case in enumerate(test_cases, start=1):
            response = backend.chat_completion(
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
                actual_arguments = parse_tool_arguments(
                    matching_tool_call.function.arguments
                )
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
        print(f"Task:            {task_name}")
        print(f"Tool:            {tool['function']['name']}")
        print(f"Should call:     {positives_passed}/{positives_total} passed")
        print(f"Should not call: {negatives_passed}/{negatives_total} passed")
        print(f"Total calls:     {passed}/{total} passed")

        if argument_total:
            print(f"Arguments:       {argument_passed}/{argument_total} passed")

        results.append(
            {
                "task": task_name,
                "tool": tool["function"]["name"],
                "should_call": {
                    "passed": positives_passed,
                    "total": positives_total,
                },
                "should_not_call": {
                    "passed": negatives_passed,
                    "total": negatives_total,
                },
                "total": {
                    "passed": passed,
                    "total": total,
                },
                "arguments": {
                    "passed": argument_passed,
                    "total": argument_total,
                },
            }
        )

    return results
