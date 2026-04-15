import mteb
import yaml
import argparse
import time

from typing import Any
from pathlib import Path
from sentence_transformers import SentenceTransformer


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data


def normalize_encode_kwargs(raw_value: Any) -> dict[str, Any]:
    if raw_value is None:
        return {}
    if isinstance(raw_value, dict):
        return raw_value
    if isinstance(raw_value, list):
        merged: dict[str, Any] = {}
        for item in raw_value:
            if not isinstance(item, dict):
                raise ValueError("Each item in encode_kwargs must be a mapping")
            merged.update(item)
        return merged
    raise ValueError("encode_kwargs must be a mapping or a list of mappings")


def build_tasks(dataset_config: dict[str, Any]) -> list[Any]:
    excluded = set(dataset_config.get("excluded_tasks", []))
    tasks = mteb.get_tasks(
        languages=dataset_config["languages"],
        modalities=dataset_config["modalities"],
        task_types=dataset_config["task_types"],
        eval_splits=["test"],
    )
    return [task for task in tasks if task.metadata.name not in excluded]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a SentenceTransformer model on MTEB tasks."
    )
    parser.add_argument(
        "--dataset-config",
        required=True,
        type=Path,
        help="Path to the dataset/task configuration YAML.",
    )
    parser.add_argument(
        "--model-config",
        required=True,
        type=Path,
        help="Path to the model configuration YAML.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device passed to SentenceTransformer.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="Optional output folder for predictions. Defaults to model_predictions--<model-name>.",
    )
    return parser.parse_args()


def main() -> None:
    bench_start = time.perf_counter()
    args = parse_args()

    dataset_config = load_yaml(args.dataset_config)
    model_config = load_yaml(args.model_config)

    tasks = build_tasks(dataset_config)

    model_name = model_config["model"]
    print(model_config.get("encode_kwargs"))
    encode_kwargs = normalize_encode_kwargs(model_config.get("encode_kwargs"))
    output_dir = args.output_dir or REPO_ROOT

    model = SentenceTransformer(model_name, device=args.device)
    cache = mteb.ResultCache(cache_path=str(output_dir))

    mteb.evaluate(
        model,
        tasks=tasks,
        cache=cache,
        encode_kwargs=encode_kwargs,
    )

    total_elapsed_seconds = time.perf_counter() - bench_start

    print(f"Benchmark completed in {total_elapsed_seconds:.2f} seconds.")

    results = cache.load_results(models=[model_name], tasks=tasks)
    df = results.to_dataframe()
    print(df)


if __name__ == "__main__":
    main()
