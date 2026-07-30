#!/usr/bin/env python3
# LEGACY: This script is a legacy evaluation script for BM25S on BEIR-style datasets. It is not part of the main MTEB evaluation pipeline and may be removed in future versions. For new evaluations, consider using the `evaluate_mteb.py` script instead.
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import mteb
from datasets import Dataset, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MTEB BM25S on a BEIR-style IR dataset, either local JSONL or Hugging Face.",
    )

    source_group = parser.add_mutually_exclusive_group(required=False)

    source_group.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=(
            "Local directory containing corpus/<split>.jsonl, "
            "queries/<split>.jsonl, qrels/<split>.jsonl."
        ),
    )

    source_group.add_argument(
        "--parquet-dir",
        type=Path,
        default=None,
        help=(
            "Local directory containing flat <split>_corpus.parquet, "
            "<split>_queries.parquet, <split>_qrels.parquet files."
        ),
    )

    source_group.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help=(
            "Hugging Face dataset repo id, e.g. lopozz/CulturaViva-Retrieval. "
            "Expected configs: corpus, queries, qrels."
        ),
    )

    parser.add_argument("--split", default="train")
    parser.add_argument("--model-name", default="mteb/baseline-bm25s")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))

    parser.add_argument(
        "--hf-corpus-config",
        default="corpus",
        help="HF dataset config name for the corpus.",
    )
    parser.add_argument(
        "--hf-queries-config",
        default="queries",
        help="HF dataset config name for the queries.",
    )
    parser.add_argument(
        "--hf-qrels-config",
        default="qrels",
        help="HF dataset config name for the qrels.",
    )

    parser.add_argument(
        "--max-corpus",
        type=int,
        default=None,
        help="Optional limit on the number of corpus documents loaded.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional limit on the number of queries loaded.",
    )
    parser.add_argument(
        "--max-qrels",
        type=int,
        default=None,
        help="Optional limit on the number of qrels loaded.",
    )

    # Optional BM25S settings, if your MTEB version/model exposes them.
    parser.add_argument(
        "--stopwords",
        default=None,
        help="Stopwords passed to MTEB BM25S, e.g. 'it', 'en', or 'none'.",
    )
    parser.add_argument(
        "--stemmer-language",
        default=None,
        help="Stemmer language passed to MTEB BM25S, e.g. 'italian', 'english', or 'none'.",
    )

    args = parser.parse_args()

    if args.dataset_dir is None and args.hf_repo_id is None and args.parquet_dir is None:
        args.dataset_dir = Path("wikinews_hard")

    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def maybe_limit(ds: Dataset, max_rows: int | None) -> Dataset:
    if max_rows is None:
        return ds
    return ds.select(range(min(max_rows, len(ds))))


def normalize_corpus_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        "id": str(row.get("_id", row.get("id"))),
        "title": str(row.get("title", "")),
        "text": str(row["text"]),
    }


def normalize_query_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        "id": str(row.get("_id", row.get("id"))),
        "text": str(row["text"]),
    }


def load_local_dataset_rows(
    dataset_dir: Path,
    split: str,
) -> tuple[Dataset, Dataset, dict[str, set[str]], str]:
    corpus_rows = read_jsonl(dataset_dir / "corpus" / f"{split}.jsonl")
    query_rows = read_jsonl(dataset_dir / "queries" / f"{split}.jsonl")
    qrel_rows = read_jsonl(dataset_dir / "qrels" / f"{split}.jsonl")

    corpus = Dataset.from_list([normalize_corpus_row(row) for row in corpus_rows])
    queries = Dataset.from_list([normalize_query_row(row) for row in query_rows])

    valid_doc_ids = set(corpus["id"])
    valid_query_ids = set(queries["id"])

    relevant_docs: dict[str, set[str]] = defaultdict(set)

    for row in qrel_rows:
        score = float(row["score"])

        if score > 0:
            query_id = str(row["query-id"])
            corpus_id = str(row["corpus-id"])

            if query_id in valid_query_ids and corpus_id in valid_doc_ids:
                relevant_docs[query_id].add(corpus_id)

    dataset_name = dataset_dir.name

    return corpus, queries, relevant_docs, dataset_name


def load_local_parquet_dataset_rows(
    dataset_dir: Path,
    split: str,
) -> tuple[Dataset, Dataset, dict[str, set[str]], str]:
    raw_corpus = Dataset.from_parquet(str(dataset_dir / f"{split}_corpus.parquet"))
    raw_queries = Dataset.from_parquet(str(dataset_dir / f"{split}_queries.parquet"))
    raw_qrels = Dataset.from_parquet(str(dataset_dir / f"{split}_qrels.parquet"))

    corpus = Dataset.from_list([normalize_corpus_row(row) for row in raw_corpus])
    queries = Dataset.from_list([normalize_query_row(row) for row in raw_queries])

    valid_doc_ids = set(corpus["id"])
    valid_query_ids = set(queries["id"])

    relevant_docs: dict[str, set[str]] = defaultdict(set)

    for row in raw_qrels:
        score = float(row["score"])

        if score > 0:
            query_id = str(row["query-id"])
            corpus_id = str(row["corpus-id"])

            if query_id in valid_query_ids and corpus_id in valid_doc_ids:
                relevant_docs[query_id].add(corpus_id)

    dataset_name = dataset_dir.name

    return corpus, queries, relevant_docs, dataset_name


def load_hf_dataset_rows(
    repo_id: str,
    split: str,
    corpus_config: str = "corpus",
    queries_config: str = "queries",
    qrels_config: str = "qrels",
) -> tuple[Dataset, Dataset, dict[str, set[str]], str]:
    raw_corpus = load_dataset(repo_id, corpus_config, split=split)
    raw_queries = load_dataset(repo_id, queries_config, split=split)
    raw_qrels = load_dataset(repo_id, qrels_config, split=split)

    corpus = Dataset.from_list([normalize_corpus_row(row) for row in raw_corpus])
    queries = Dataset.from_list([normalize_query_row(row) for row in raw_queries])

    valid_doc_ids = set(corpus["id"])
    valid_query_ids = set(queries["id"])

    relevant_docs: dict[str, set[str]] = defaultdict(set)

    for row in raw_qrels:
        score = float(row.get("score", 0))

        if score > 0:
            print(row)
            query_id = str(row["query-id"])
            corpus_id = str(row["corpus-id"])

            if query_id in valid_query_ids and corpus_id in valid_doc_ids:
                relevant_docs[query_id].add(corpus_id)

    dataset_name = repo_id.replace("/", "__")

    return corpus, queries, relevant_docs, dataset_name


def ranked_doc_ids(results: dict[str, float]) -> list[str]:
    return [
        doc_id
        for doc_id, _ in sorted(results.items(), key=lambda item: item[1], reverse=True)
    ]


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:k], start=1)
        if doc_id in relevant
    )

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg else 0.0


def evaluate_predictions(
    predictions: dict[str, dict[str, float]],
    relevant_docs: dict[str, set[str]],
) -> dict[str, float]:
    ks = [10]

    sums: dict[str, float] = defaultdict(float)
    evaluated = 0

    for query_id, relevant in relevant_docs.items():
        ranked = ranked_doc_ids(predictions.get(query_id, {}))

        for k in ks:
            sums[f"ndcg_at_{k}"] += ndcg_at_k(ranked, relevant, k)

        evaluated += 1

    if evaluated == 0:
        raise ValueError("No positive qrels found after filtering against loaded corpus/queries.")

    return {
        metric: value / evaluated
        for metric, value in sorted(sums.items())
    }


def get_bm25s_model(args: argparse.Namespace):
    kwargs = {}

    if args.stopwords is not None:
        kwargs["stopwords"] = args.stopwords

    if args.stemmer_language is not None:
        kwargs["stemmer_language"] = args.stemmer_language

    try:
        return mteb.get_model(args.model_name, **kwargs)
    except TypeError:
        if kwargs:
            print(
                "Warning: this MTEB model/get_model version did not accept "
                f"BM25 kwargs {kwargs}. Retrying without them."
            )
        return mteb.get_model(args.model_name)


def main() -> None:
    args = parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be a positive integer.")

    if args.hf_repo_id is not None:
        corpus, queries, relevant_docs, dataset_name = load_hf_dataset_rows(
            repo_id=args.hf_repo_id,
            split=args.split,
            corpus_config=args.hf_corpus_config,
            queries_config=args.hf_queries_config,
            qrels_config=args.hf_qrels_config,
        )
        dataset_source = {
            "type": "huggingface",
            "repo_id": args.hf_repo_id,
            "corpus_config": args.hf_corpus_config,
            "queries_config": args.hf_queries_config,
            "qrels_config": args.hf_qrels_config,
        }
    elif args.parquet_dir is not None:
        corpus, queries, relevant_docs, dataset_name = load_local_parquet_dataset_rows(
            dataset_dir=args.parquet_dir,
            split=args.split,
        )
        dataset_source = {
            "type": "local_parquet",
            "dataset_dir": str(args.parquet_dir),
        }
    else:
        corpus, queries, relevant_docs, dataset_name = load_local_dataset_rows(
            dataset_dir=args.dataset_dir,
            split=args.split,
        )
        dataset_source = {
            "type": "local",
            "dataset_dir": str(args.dataset_dir),
        }

    model = get_bm25s_model(args)

    start = time.perf_counter()

    model.index(
        corpus,
        task_metadata=None,
        hf_split=args.split,
        hf_subset=dataset_name,
        encode_kwargs={},
    )

    predictions = model.search(
        queries,
        task_metadata=None,
        hf_split=args.split,
        hf_subset=dataset_name,
        top_k=args.top_k,
        encode_kwargs={},
    )

    evaluation_time = time.perf_counter() - start

    scores = evaluate_predictions(predictions, relevant_docs)
    scores["main_score"] = scores["ndcg_at_10"]

    safe_model_name = args.model_name.replace("/", "__")
    result_dir = args.output_dir / safe_model_name
    result_dir.mkdir(parents=True, exist_ok=True)

    result_path = result_dir / f"{dataset_name}_{args.split}.json"

    payload = {
        "model_name": args.model_name,
        "dataset_source": dataset_source,
        "split": args.split,
        "top_k": args.top_k,
        "num_corpus": len(corpus),
        "num_queries": len(queries),
        "num_qrels": sum(len(doc_ids) for doc_ids in relevant_docs.values()),
        "evaluation_time": evaluation_time,
        "scores": scores,
        "settings": {
            "stopwords": args.stopwords,
            "stemmer_language": args.stemmer_language,
        },
    }

    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(scores, indent=2))
    print(f"Saved result to: {result_path}")


if __name__ == "__main__":
    main()