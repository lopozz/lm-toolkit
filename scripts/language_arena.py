"""Make two LLMs converse with each other and watch the exchange live.

Each speaker (A/B) has its own model and system prompt and is only ever shown
the conversation from its own point of view: its own past replies appear as
"assistant" turns, the other speaker's replies as "user" turns. Both models
are queried through the same OpenAI-compatible /chat/completions endpoint.

Useful for probing how a model behaves in an open-ended dialogue, comparing
two models/personas against each other, or generating synthetic multi-turn
conversation data.

Example:
    python3 scripts/language_arena.py \
      --base-url http://localhost:8000/v1 \
      --model-a mistralai/Ministral-3-3B-Instruct-2512 \
      --model-b mistralai/Ministral-3-3B-Instruct-2512 \
      --scenario philosophy \
      --opener "What is the meaning of work?"
"""

import argparse
import json
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from lm_toolkit.chats import stream_chat_completion


def run_arena(
    base_url: str,
    api_key: str,
    model_a: str,
    system_a: str,
    model_b: str,
    system_b: str,
    opener: str,
    turns: int,
    temperature: float,
    max_tokens: int,
) -> list[dict]:
    # Each speaker sees the conversation from its own point of view: its own
    # previous replies are "assistant" turns, the other speaker's replies are
    # folded in as "user" turns. Only the transcript (role "a"/"b" + text) is
    # kept as the source of truth; the two message lists are derived from it
    # on every turn.
    transcript = [{"speaker": "a", "text": opener}]

    console = Console()
    styles = {"a": "cyan", "b": "magenta"}
    console.print(Panel(opener, title=f"A · {model_a}", border_style=styles["a"]))

    client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")

    for turn in range(turns):
        speaker = "b" if turn % 2 == 0 else "a"
        model = model_b if speaker == "b" else model_a
        system_prompt = system_b if speaker == "b" else system_a

        messages = [{"role": "system", "content": system_prompt}]
        for entry in transcript:
            role = "assistant" if entry["speaker"] == speaker else "user"
            messages.append({"role": role, "content": entry["text"]})

        title = f"{speaker.upper()} · {model}"
        accumulated = ""
        with Live(Panel("", title=title, border_style=styles[speaker]), console=console, refresh_per_second=15) as live:
            def on_delta(delta: str) -> None:
                nonlocal accumulated
                accumulated += delta
                live.update(Panel(accumulated, title=title, border_style=styles[speaker]))

            reply = stream_chat_completion(
                client=client,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                on_delta=on_delta,
            )

        transcript.append({"speaker": speaker, "text": reply})

    return transcript


CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "arena"


def load_scenario_prompts(scenario: str) -> tuple[str, str]:
    scenario_dir = CONFIGS_DIR / scenario

    system_a_path = scenario_dir / "system_a.txt"
    system_b_path = scenario_dir / "system_b.txt"

    missing = [path for path in (system_a_path, system_b_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Scenario {scenario!r} is missing: {', '.join(str(path) for path in missing)}"
        )

    return system_a_path.read_text(encoding="utf-8").strip(), system_b_path.read_text(encoding="utf-8").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two LLMs talk to each other over an OpenAI-compatible endpoint.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI-compatible API base URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--api-key", default="", help="Bearer token for the endpoint, if required")
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--scenario", help=f"Name of a scenario dir under {CONFIGS_DIR} containing system_a.txt and system_b.txt")
    parser.add_argument("--system-a", help="System prompt for speaker A (overrides --scenario if both given)")
    parser.add_argument("--system-b", help="System prompt for speaker B (overrides --scenario if both given)")
    parser.add_argument("--opener", required=True, help="First message, sent as speaker A's opening line")
    parser.add_argument("--turns", type=int, default=10, help="Number of replies to generate (alternating A/B)")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path, default=None, help="Path to write the transcript as JSONL; defaults to transcripts/<timestamp>.jsonl")
    args = parser.parse_args()

    if not args.scenario and not (args.system_a and args.system_b):
        parser.error("either --scenario or both --system-a and --system-b are required")

    if args.scenario:
        scenario_system_a, scenario_system_b = load_scenario_prompts(args.scenario)
        args.system_a = args.system_a or scenario_system_a
        args.system_b = args.system_b or scenario_system_b

    return args


def main() -> None:
    args = parse_args()

    transcript = run_arena(
        base_url=args.base_url,
        api_key=args.api_key,
        model_a=args.model_a,
        system_a=args.system_a,
        model_b=args.model_b,
        system_b=args.system_b,
        opener=args.opener,
        turns=args.turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    if args.output is None:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as file:
        for entry in transcript:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nSaved transcript to: {args.output}")


if __name__ == "__main__":
    main()
