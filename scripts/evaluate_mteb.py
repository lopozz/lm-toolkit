"""
Evaluate retrieval models with a temporary sparse-model workaround.

This script keeps a single evaluation entry point for different retrieval model
families while MTEB does not yet provide full native support for
SentenceTransformers `SparseEncoder` models.

For dense embedding models and MTEB-supported baselines such as BM25, the script
uses the standard MTEB flow:

    model = mteb.get_model(model_name)
    results = mteb.evaluate(...)

For sparse SPLADE-style models, the script uses SentenceTransformers'
`SparseInformationRetrievalEvaluator`, then writes the output into the same
project result directory used by MTEB:

    ../results/<organization>__<model>/prediction_folder/

The sparse branch also saves a small MTEB-like JSON result file so that sparse
and dense evaluations can be tracked in a similar way until MTEB supports
`SparseEncoder` directly.

Intent
------
The goal is not to replace MTEB, but to bridge the current gap between:

1. dense/BM25 models, which can be evaluated directly with `mteb.evaluate`, and
2. sparse encoders, which currently require a sparse-specific evaluator.

This makes it possible to compare dense, BM25, and SPLADE-style sparse retrieval
models in the same experiment structure, with consistent output folders and
similar result metadata.

Current limitations
-------------------
The sparse path is a workaround. It creates MTEB-like JSON files, but not real
MTEB `TaskResult` objects. The scores are produced by SentenceTransformers'
sparse evaluator, so metric names and output details may differ from native
MTEB output.

The sparse path currently assumes retrieval datasets with explicit corpus,
queries, and qrels subsets, for example:

    mteb/MuPLeR-retrieval
    it-corpus
    it-queries
    it-qrels

Additional tasks may require adding their dataset subset names and ID-column
conventions.

Future development
------------------
When MTEB adds native support for SentenceTransformers `SparseEncoder` models,
the sparse branch should be removable. At that point, sparse models should be
evaluated through the same standard MTEB API as dense models:

    model = SparseEncoder(model_name)

    results = mteb.evaluate(
        model=model,
        tasks=tasks,
        cache=cache,
        encode_kwargs={"batch_size": batch_size},
        prediction_folder=prediction_folder,
        overwrite_strategy="always",
    )

Until then, this script provides a reproducible compatibility layer for sparse
retrieval evaluation while preserving a directory and result structure close to
the standard MTEB workflow.
"""

import argparse
import json
import time
import yaml
import mteb

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from datasets import load_dataset

from mteb.models import ModelMeta
from mteb.models.model_meta import ScoringFunction
from sentence_transformers import SparseEncoder, SparseEncoderModelCardData
from sentence_transformers.sparse_encoder.evaluation import (
    SparseInformationRetrievalEvaluator,
)
from sentence_transformers.sentence_transformer.modules import Router, Transformer
from sentence_transformers.sparse_encoder.modules import (
    SpladePooling,
    SparseStaticEmbedding,
)


SPARSE_MODELS = {
    "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
    "nickprock/splade-bert-base-italian-xxl-uncased-cv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval models with MTEB.")
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name or Hugging Face repository.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("./configs/ds/mteg-retrieval-ita.yml"),
        help="Dataset task configuration YAML.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite-strategy", default="always")
    parser.add_argument("--results-dir", type=Path, default=Path("./results"))
    parser.add_argument(
        "--no-query-expansion",
        action="store_false",
        dest="query_expansion",
        help="Use lexical query tokens instead of SPLADE query expansion.",
    )
    parser.add_argument(
        "--task-limit",
        type=int,
        default=1,
        help="Maximum number of matching tasks to evaluate. Use -1 to evaluate all tasks.",
    )
    return parser.parse_args()


def reshape_sparse_predictions(
    source_path: Path,
    destination_path: Path,
    model_name: str,
    model_revision: str,
    hf_subset: str,
    split: str,
) -> None:
    predictions = {}

    with source_path.open("r") as source:
        for line in source:
            row = json.loads(line)
            predictions[str(row["query_id"])] = {
                str(result["corpus_id"]): result["score"] for result in row["results"]
            }

    mteb_predictions = {
        "mteb_model_meta": {
            "model_name": model_name,
            "revision": model_revision,
        },
        hf_subset: {
            split: predictions,
        },
    }

    with destination_path.open("w") as destination:
        json.dump(mteb_predictions, destination)

    source_path.unlink()


def get_package_versions() -> dict[str, str | None]:
    package_versions = {}

    for package_name in [
        "mteb",
        "torch",
        "sentence-transformers",
        "flash-attn",
        "transformers",
    ]:
        try:
            package_versions[package_name] = version(package_name)
        except PackageNotFoundError:
            package_versions[package_name] = None

    return package_versions


def save_sparse_model_metadata(
    result_folder: Path,
    model: SparseEncoder,
    model_name: str,
    model_revision: str,
    languages: list[str],
) -> None:
    model_meta = ModelMeta.create_empty(
        {
            "name": model_name,
            "revision": model_revision,
            "languages": languages,
            "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "license": model.model_card_data.license,
            "open_weights": True,
            "framework": ["Sentence Transformers", "Transformers", "safetensors"],
            "reference": f"https://huggingface.co/{model_name}",
            "similarity_fn_name": ScoringFunction.DOT_PRODUCT,
            "model_type": ["sparse"],
        }
    )

    with (result_folder / "model_meta.json").open("w") as metadata_file:
        json.dump(model_meta.to_dict(), metadata_file, default=str, indent=4)


def save_run_settings(
    result_folder: Path,
    task_name: str,
    split: str,
    hf_subset: str,
    batch_size: int,
) -> None:
    run_settings_path = result_folder / "run_settings.jsonl"
    run_settings = {
        "task": task_name,
        "split": split,
        "subset": hf_subset,
        "version": get_package_versions(),
        "encode_kwargs": {"batch_size": batch_size},
    }
    existing_settings = []

    if run_settings_path.exists():
        with run_settings_path.open("r") as settings_file:
            existing_settings = [
                json.loads(line) for line in settings_file if line.strip()
            ]

    run_key = (task_name, split, hf_subset)
    existing_settings = [
        setting
        for setting in existing_settings
        if (setting["task"], setting["split"], setting["subset"]) != run_key
    ]

    with run_settings_path.open("w") as settings_file:
        for setting in [*existing_settings, run_settings]:
            settings_file.write(json.dumps(setting) + "\n")


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be a positive integer.")
    if args.task_limit != -1 and args.task_limit < 1:
        raise ValueError("--task-limit must be -1 or a positive integer.")

    model_name = args.model_name
    batch_size = args.batch_size
    overwrite_strategy = args.overwrite_strategy
    cache = mteb.ResultCache(".")
    safe_model_name = model_name.replace("/", "__")

    # Load MTEB tasks

    with args.config_path.open("r") as f:
        config = yaml.safe_load(f)

    excluded = set(config["excluded_tasks"])

    tasks = mteb.get_tasks(
        languages=config["languages"],
        modalities=config["modalities"],
        task_types=config["task_types"],
        exclusive_language_filter=True,
        eval_splits=["test"],
    )

    tasks = [t for t in tasks if t.metadata.name not in excluded]

    if args.task_limit != -1:
        tasks = tasks[: args.task_limit]

    if model_name not in SPARSE_MODELS:
        model = mteb.get_model(model_name)
        prediction_folder = args.results_dir / safe_model_name / "prediction_folder"
        prediction_folder.mkdir(parents=True, exist_ok=True)

        results = mteb.evaluate(
            model=model,
            tasks=tasks,
            cache=cache,
            encode_kwargs={"batch_size": batch_size},
            prediction_folder=str(prediction_folder),
            overwrite_strategy=overwrite_strategy,
        )

        print(results)

    else:
        mlm_transformer = Transformer(
            model_name,
            transformer_task="fill-mask",
        )

        splade_pooling = SpladePooling(
            pooling_strategy="max",
            embedding_dimension=mlm_transformer.get_embedding_dimension(),
        )

        query_modules = (
            [mlm_transformer, splade_pooling]
            if args.query_expansion
            else [
                SparseStaticEmbedding(
                    tokenizer=mlm_transformer.tokenizer,
                    frozen=True,
                )
            ]
        )
        router = Router.for_query_document(
            query_modules=query_modules,
            document_modules=[mlm_transformer, splade_pooling],
        )

        model = SparseEncoder(
            modules=[router],
            model_card_data=SparseEncoderModelCardData(
                model_name=model_name,
            ),
        )
        all_task_results = []

        model_revision = mlm_transformer.auto_model.config._commit_hash
        result_folder = args.results_dir / safe_model_name / model_revision
        result_folder.mkdir(parents=True, exist_ok=True)
        prediction_folder = args.results_dir / safe_model_name / "prediction_folder"
        prediction_folder.mkdir(parents=True, exist_ok=True)
        evaluated_languages = sorted(
            {
                language
                for task in tasks
                for hf_subset in task.hf_subsets
                for language in task.metadata.eval_langs[hf_subset]
            }
        )
        save_sparse_model_metadata(
            result_folder=result_folder,
            model=model,
            model_name=model_name,
            model_revision=model_revision,
            languages=evaluated_languages,
        )

        for task in tasks:
            task_name = task.metadata.name
            split = "test"

            print(f"Evaluating sparse model on task: {task_name}")

            # Dataset mapping
            # Add more tasks here as needed.

            if task_name == "MuPLeR-retrieval":
                dataset_path = "mteb/MuPLeR-retrieval"
                hf_subset = "it"
                corpus_name = "it-corpus"
                queries_name = "it-queries"
                qrels_name = "it-qrels"
                id_column = "id"

            else:
                raise ValueError(
                    f"No sparse dataset mapping defined for task: {task_name}. "
                    "Add dataset_path, corpus_name, queries_name, qrels_name, and id_column."
                )

            corpus_ds = load_dataset(
                path=dataset_path,
                name=corpus_name,
                split=split,
            )
            queries_ds = load_dataset(
                path=dataset_path,
                name=queries_name,
                split=split,
            )
            qrels_ds = load_dataset(
                path=dataset_path,
                name=qrels_name,
                split=split,
            )

            queries = {str(row[id_column]): row["text"] for row in queries_ds}

            corpus = {str(row[id_column]): row["text"] for row in corpus_ds}

            relevant_docs = {}

            for row in qrels_ds:
                qid = str(row["query-id"])
                cid = str(row["corpus-id"])
                score = row["score"]

                if score > 0:
                    relevant_docs.setdefault(qid, set()).add(cid)

            ir_evaluator = SparseInformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                show_progress_bar=True,
                batch_size=batch_size,
                accuracy_at_k=[1],
                precision_recall_at_k=[1, 3, 5, 10],
                mrr_at_k=[10],
                ndcg_at_k=[10],
                map_at_k=[100],
                write_csv=False,
                write_predictions=True,
            )

            evaluation_start = time.perf_counter()
            sparse_scores = ir_evaluator(
                model,
                output_path=str(prediction_folder),
            )
            evaluation_time = time.perf_counter() - evaluation_start

            sparse_predictions_path = (
                prediction_folder
                / "Information-Retrieval_evaluation_predictions_dot.jsonl"
            )
            mteb_predictions_path = prediction_folder / f"{task_name}_predictions.json"
            reshape_sparse_predictions(
                source_path=sparse_predictions_path,
                destination_path=mteb_predictions_path,
                model_name=model_name,
                model_revision=model_revision,
                hf_subset=hf_subset,
                split=split,
            )

            # Convert SentenceTransformers sparse output to MTEB-like output
            mteb_scores = {
                "ndcg_at_10": sparse_scores.get("dot_ndcg@10"),
                "map_at_100": sparse_scores.get("dot_map@100"),
                "recall_at_1": sparse_scores.get("dot_recall@1"),
                "recall_at_3": sparse_scores.get("dot_recall@3"),
                "recall_at_5": sparse_scores.get("dot_recall@5"),
                "recall_at_10": sparse_scores.get("dot_recall@10"),
                "accuracy": sparse_scores.get("dot_accuracy@1"),
                "precision_at_1": sparse_scores.get("dot_precision@1"),
                "precision_at_3": sparse_scores.get("dot_precision@3"),
                "precision_at_5": sparse_scores.get("dot_precision@5"),
                "precision_at_10": sparse_scores.get("dot_precision@10"),
                "mrr_at_10": sparse_scores.get("dot_mrr@10"),
                # MTEB retrieval tasks usually use nDCG@10 as main score.
                "main_score": sparse_scores.get("dot_ndcg@10"),
                "hf_subset": hf_subset,
                "languages": task.metadata.eval_langs[hf_subset],
                "mteb_version": mteb.__version__,
            }

            task_result = {
                "dataset_revision": task.metadata.revision,
                "task_name": task_name,
                "mteb_version": mteb.__version__,
                "scores": {
                    split: [mteb_scores],
                },
                "evaluation_time": evaluation_time,
                "kg_co2_emissions": None,
                "date": time.time(),
            }

            all_task_results.append(task_result)

            task_result_path = result_folder / f"{task_name}.json"

            with open(task_result_path, "w") as f:
                json.dump(task_result, f, indent=2)

            save_run_settings(
                result_folder=result_folder,
                task_name=task_name,
                split=split,
                hf_subset=hf_subset,
                batch_size=batch_size,
            )

            print(f"Saved task result to: {task_result_path}")


if __name__ == "__main__":
    main()
