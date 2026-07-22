"""
Evaluate the activation behavior of a SPLADE-style sparse encoder over a full
retrieval dataset.

Unlike evaluate_mteb.py, which measures ranking quality (nDCG, MAP, recall),
this script measures how a SPLADE-style model represents queries and documents:

- active_dims / sparsity ratio: Number of vocabulary dimensions with non-zero
  activation, and the complementary fraction of the embedding that stays at
  zero (via SparseEncoder.sparsity()).
- Expansion ratio: Activated dimensions relative to the number of input tokens.
- Expansion weight mass: Total activation weight assigned to terms not present
  in the original input.
- Lexical retention: Proportion of original input terms retained with non-zero
  activation.

Each task describes one MTEB-style retrieval dataset subset:

    {"task_name": "MuPLeR-retrieval", "language": "it", "split": "test"}

For each task, every query and every document in the corpus is encoded, and
per-example metrics are aggregated into summary statistics (mean, median,
p90).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from rich.progress import track
from sentence_transformers import SparseEncoder

Expansion = list[dict[str, Any]]


@dataclass(frozen=True)
class RetrievalTask:
    task_name: str
    language: str = "it"
    split: str = "test"


def sort_expansion(expansion: Expansion) -> Expansion:
    expansion.sort(key=lambda item: (-float(item["weight"]), str(item["token"])))
    return expansion


def tensor_expansion(vector: Any, vocab: dict[int, str]) -> Expansion:
    if not torch.is_tensor(vector):
        vector = torch.as_tensor(vector)

    if vector.is_sparse:
        vector = vector.coalesce()
        indices = vector.indices()
        values = vector.values()
        if indices.ndim == 2:
            indices = indices[-1]
        active = zip(indices.tolist(), values.tolist(), strict=False)
    else:
        active_indices = torch.nonzero(vector > 0, as_tuple=False).flatten()
        active = zip(active_indices.tolist(), vector[active_indices].tolist(), strict=False)

    expansion = [
        {"token": vocab.get(int(index), f"[UNK_{index}]"), "weight": float(weight)}
        for index, weight in active
        if float(weight) > 0
    ]
    return sort_expansion(expansion)


def input_terms(text: str, tokenizer: Any) -> set[str]:
    """Tokenize `text` with the model's own tokenizer so lexical comparisons
    line up with the vocabulary the expansion tokens are drawn from."""
    return set(tokenizer.tokenize(text))


def active_dims(expansion: Expansion) -> int:
    return len(expansion)


def expansion_ratio(expansion: Expansion, terms: set[str]) -> float:
    """Fraction of active terms that do not appear in the input."""
    if not expansion:
        return 0.0
    novel = sum(1 for item in expansion if item["token"] not in terms)
    return novel / len(expansion)


def expansion_weight_ratio(expansion: Expansion, terms: set[str]) -> float:
    """Fraction of total activation weight carried by novel (expansion) terms."""
    total_weight = sum(float(item["weight"]) for item in expansion)
    if total_weight == 0:
        return 0.0
    novel_weight = sum(
        float(item["weight"]) for item in expansion if item["token"] not in terms
    )
    return novel_weight / total_weight


def lexical_retention(expansion: Expansion, terms: set[str]) -> tuple[float, float]:
    """Returns (retention by count, retention by weight mass) of input terms
    that remain active in the expansion."""
    if not terms:
        return 0.0, 0.0

    active_terms = {item["token"] for item in expansion}
    retained = terms & active_terms
    retention_count = len(retained) / len(terms)

    total_weight = sum(float(item["weight"]) for item in expansion)
    if total_weight == 0:
        return retention_count, 0.0
    retained_weight = sum(
        float(item["weight"]) for item in expansion if item["token"] in retained
    )
    return retention_count, retained_weight / total_weight


def load_task_texts(task: RetrievalTask) -> tuple[dict[str, str], dict[str, str]]:
    # WebFAQRetrieval uses ISO 639-3 subset names (e.g. "ita-corpus") while
    # other MTEB retrieval tasks used here use ISO 639-1 (e.g. "it-corpus").
    lang_prefix = "ita" if task.task_name == "WebFAQRetrieval" else task.language
    params = {"path": f"mteb/{task.task_name}", "split": task.split}
    corpus_ds = load_dataset(name=f"{lang_prefix}-corpus", **params)
    queries_ds = load_dataset(name=f"{lang_prefix}-queries", **params)
    corpus = {str(row["id"]): str(row["text"]) for row in corpus_ds}
    queries = {str(row["id"]): str(row["text"]) for row in queries_ds}
    return corpus, queries


def encode_expansions(
    texts: dict[str, str],
    model: SparseEncoder,
    vocab: dict[int, str],
    mode: str,
    batch_size: int,
) -> tuple[dict[str, Expansion], dict[str, float]]:
    text_ids = list(texts)
    encoded: dict[str, Expansion] = {}
    sparsity_ratios: dict[str, float] = {}

    for start in track(
        range(0, len(text_ids), batch_size),
        description=f"[sparse_retrieval] encoding {mode}",
    ):
        batch_ids = text_ids[start : start + batch_size]
        batch_texts = [texts[text_id] for text_id in batch_ids]
        if mode == "query":
            vectors = model.encode_query(batch_texts, convert_to_sparse_tensor=False)
        else:
            vectors = model.encode_document(batch_texts, convert_to_sparse_tensor=False)

        for text_id, vector in zip(batch_ids, vectors, strict=True):
            encoded[text_id] = tensor_expansion(vector, vocab)
            sparsity_ratios[text_id] = model.sparsity(vector)["sparsity_ratio"]

    return encoded, sparsity_ratios


def metrics(expansion: Expansion, terms: set[str], sparsity_ratio: float) -> dict[str, float]:
    retention_count, retention_weight = lexical_retention(expansion, terms)
    return {
        "active_dims": active_dims(expansion),
        "sparsity_ratio": sparsity_ratio,
        "expansion_ratio": expansion_ratio(expansion, terms),
        "expansion_weight_ratio": expansion_weight_ratio(expansion, terms),
        "lexical_retention_count": retention_count,
        "lexical_retention_weight": retention_weight,
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = fraction * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def aggregate_metrics(per_example: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not per_example:
        return {}

    metric_names = per_example[0].keys()
    aggregated: dict[str, dict[str, float]] = {}

    for metric_name in metric_names:
        values = [example[metric_name] for example in per_example]
        aggregated[metric_name] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p90": percentile(values, 0.9),
        }

    return aggregated


def print_summary(task_name: str, model_name: str, stats: dict[str, Any]) -> None:
    print()
    print("Summary")
    print(f"Task:  {task_name}")
    print(f"Model: {model_name}")

    for group_name in ("query_stats", "document_stats"):
        print(f"{group_name}:")
        for metric_name, values in stats[group_name].items():
            print(
                f"  {metric_name:<26} mean={values['mean']:.3f} "
                f"median={values['median']:.3f} p90={values['p90']:.3f}"
            )


def evaluate_sparse_retrieval(
    model: str,
    tasks: list[dict[str, Any]],
    backend: SparseEncoder,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    backend.eval()
    vocab = {index: token for token, index in backend.tokenizer.get_vocab().items()}

    results: list[dict[str, Any]] = []

    for task_config in tasks:
        task = RetrievalTask(**task_config)
        corpus, queries = load_task_texts(task)

        query_ids = list(queries)
        document_ids = list(corpus)
        selected_queries = {query_id: queries[query_id] for query_id in query_ids}
        selected_documents = {doc_id: corpus[doc_id] for doc_id in document_ids}

        query_expansions, query_sparsity_ratios = encode_expansions(
            selected_queries, backend, vocab, mode="query", batch_size=batch_size
        )
        document_expansions, document_sparsity_ratios = encode_expansions(
            selected_documents, backend, vocab, mode="document", batch_size=batch_size
        )

        query_metrics = [
            metrics(
                query_expansions[query_id],
                input_terms(selected_queries[query_id], backend.tokenizer),
                query_sparsity_ratios[query_id],
            )
            for query_id in query_ids
        ]
        document_metrics = [
            metrics(
                document_expansions[doc_id],
                input_terms(selected_documents[doc_id], backend.tokenizer),
                document_sparsity_ratios[doc_id],
            )
            for doc_id in document_ids
        ]

        stats = {
            "query_stats": aggregate_metrics(query_metrics),
            "document_stats": aggregate_metrics(document_metrics),
        }
        print_summary(task.task_name, model, stats)

        results.append(
            {
                "task": task.task_name,
                "model": model,
                "num_queries": len(query_ids),
                "num_documents": len(document_ids),
                **stats,
            }
        )

    return results
