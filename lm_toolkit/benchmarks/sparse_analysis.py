from __future__ import annotations

import math
import json
import torch

import pandas as pd
import matplotlib.pyplot as plt

from typing import Any
from pathlib import Path
from rich.progress import track
from scipy.stats import spearmanr
from datasets import load_dataset
from dataclasses import dataclass
from collections import defaultdict
from sentence_transformers import SparseEncoder

DOCUMENT_METRICS = (
    "active_dims",
    "sparsity_ratio",
    "expansion_ratio",
    "expansion_weight_ratio",
    "retention_ratio",
    "retention_weight_ratio",
    "retention",
    "expansion",
)

FIXED_LENGTH_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("0-15", 0, 15),
    ("16-25", 16, 25),
    ("26-40", 26, 40),
    ("41-60", 41, 60),
    ("61-90", 61, 90),
    ("91-150", 91, 150),
    (">150", 151, None),
)

DEFAULT_TASK_NAMES = (
    "CulturaViva-Retrieval",
    "MuPLeR-retrieval",
    "WikipediaRetrievalMultilingual",
    "WebFAQRetrieval",
)

Expansion = list[dict[int, Any]]

@dataclass(frozen=True)
class RetrievalTask:
    task_name: str
    language: str = "it"
    split: str = "test"


def load_corpus(task: RetrievalTask) -> dict[str, str]:
    lang_prefix = (
        "ita" if task.task_name == "WebFAQRetrieval" else task.language
    )
    id_column = (
        "_id"
        if task.task_name == "WikipediaRetrievalMultilingual"
        else "id"
    )

    if task.task_name == "CulturaViva-Retrieval":
        corpus_ds = load_dataset(
            path="lopozz/CulturaViva-Retrieval",
            name="corpus",
            split=task.split,
        )
    else:
        corpus_ds = load_dataset(
            path=f"mteb/{task.task_name}",
            name=f"{lang_prefix}-corpus",
            split=task.split,
        )

    return {
        str(row[id_column]): str(row["text"])
        for row in corpus_ds
    }

def tensor_expansion(vector: Any) -> Expansion:
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
        active = zip(
            active_indices.tolist(),
            vector[active_indices].tolist(),
            strict=False,
        )

    expansion = [
        {
            "token": int(index),
            "weight": float(weight),
        }
        for index, weight in active
        if float(weight) > 0
    ]
    expansion.sort(key=lambda item: (-float(item["weight"]), str(item["token"])))
    return expansion

def lexical_retention(
    expansion: Expansion,
    terms: set[int],
) -> tuple[int, float, float]:
    """Returns (absolute retained count, retention ratio, retention weight ratio)."""
    active_terms = {item["token"] for item in expansion}
    retained = terms & active_terms

    retention_ratio = len(retained) / len(terms)

    total_weight = sum(float(item["weight"]) for item in expansion)

    if total_weight == 0:
        return len(retained), retention_ratio, 0.0

    retained_weight = sum(
        float(item["weight"]) for item in expansion if item["token"] in retained
    )
    return len(retained), retention_ratio, retained_weight / total_weight

def expansion_ratio(expansion: Expansion, terms: set[int]) -> float:
    """Fraction of active dimensions that are not literal input terms."""
    if not expansion:
        return 0.0

    novel = sum(1 for item in expansion if item["token"] not in terms)
    return novel / len(expansion)

def expansion_weight_ratio(expansion: Expansion, terms: set[int]) -> float:
    """Fraction of activation mass assigned to non-input terms."""
    total_weight = sum(float(item["weight"]) for item in expansion)

    if total_weight == 0:
        return 0.0

    novel_weight = sum(
        float(item["weight"]) for item in expansion if item["token"] not in terms
    )
    return novel_weight / total_weight

def document_metrics(
    text: str,
    tokenizer: Any,
    encoder_max_length: int,
    vector: Any,
    sparsity_ratio: float,
) -> dict[str, float]:
    """
    Tokenizes `text` once to derive both the length stats (raw/effective/
    model-input length, whether it was truncated) and the `terms` set used
    for the expansion/retention metrics -- so both are guaranteed to reflect
    the same truncated view of the document the encoder actually saw.
    """
    expansion = tensor_expansion(vector)

    raw_ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    raw_length = len(raw_ids)

    effective_ids = raw_ids[:encoder_max_length - 2]
    was_truncated = int(raw_length > encoder_max_length - 2)

    assert effective_ids, f"Tokenizer returned an empty token list within encoder_max_length for text: {text!r}"

    terms = set(effective_ids)

    retention, retention_ratio, retention_weight_ratio = lexical_retention(
        expansion, terms
    )
    active_dims = len(expansion)

    return {
        
        "raw_token_length": raw_length,                                                 # the document's true length
        "effective_token_length": len(effective_ids),                                   # token after truncation (minus special tokens).
        "was_truncated": was_truncated,                                                 # if raw_token_length exceeded the encoder's capacity
        "active_dims": float(active_dims),                                              # number of non-zero dimensions in the sparse vector
        "sparsity_ratio": float(sparsity_ratio),                                        # fraction of the vocabulary that stayed at zero (1 - density).
        "expansion_ratio": expansion_ratio(expansion, terms),                           # fraction of active dims that are NOT literal input terms (the real expansion).
        "retention_ratio": retention_ratio,                                             # input terms that stayed active
        "expansion_weight_ratio": expansion_weight_ratio(expansion, terms),             # fraction of total activation weight carried by those expansion terms
        "retention_weight_ratio": retention_weight_ratio,                               # same, but weighted by how much activation weight those terms carry.
        "retention": retention,                                                         # count of active dims that are literal input terms
        "expansion": active_dims - retention,                                           # count of active dims that are expansion terms
    }

def encode_document_rows(
    task: RetrievalTask,
    corpus: dict[str, str],
    backend: SparseEncoder,
    batch_size: int,
) -> list[dict[str, Any]]:
    document_ids = list(corpus)

    tokenizer = backend.tokenizer
    encoder_max_length = getattr(backend, "max_seq_length", None) or getattr(tokenizer, "model_max_length", None)

    assert encoder_max_length is not None, "Could not determine encoder max length from model or tokenizer."

    print(
        f"\nEncoding {task.task_name}: {len(document_ids):,} documents; "
        f"encoder_max_length={encoder_max_length}"
    )

    rows: list[dict[str, Any]] = []

    # encode_document's own batch_size only controls forward-pass batching; it
    # still returns one stacked tensor covering everything passed in. Batching
    # here bounds peak memory to one chunk's dense [batch_size, vocab_size]
    # output at a time, converting each to the compact Expansion representation
    # before moving on, instead of materializing the whole corpus at once.
    for start in track(
        range(0, len(document_ids), batch_size),
        description=f"[length-density] {task.task_name}",
    ):
        batch_ids = document_ids[start : start + batch_size]
        batch_texts = [corpus[document_id] for document_id in batch_ids]

        vectors = backend.encode_document(batch_texts, convert_to_sparse_tensor=False) # [B, V]

        for document_id, text, vector in zip(batch_ids, batch_texts, vectors, strict=True):
            sparsity = backend.sparsity(vector)["sparsity_ratio"]

            row = {
                "dataset": task.task_name,
                "document_id": document_id,
                **document_metrics(text, tokenizer, encoder_max_length, vector, sparsity),
            }
            rows.append(row)

    return rows

def add_dataset_quantile_buckets(
    frame: pd.DataFrame,
    num_buckets: int,
    fixed_buckets: tuple[tuple[str, int, int | None], ...] | None = None,
) -> pd.DataFrame:
    """
    Assigns each document to a length bucket, written to a single "bucket"
    column either way.

    If fixed_buckets is given, uses those fixed absolute-length ranges
    (comparable across datasets/models, e.g. FIXED_LENGTH_BUCKETS). Otherwise
    falls back to num_buckets equal-frequency quantile buckets ("Q1".."Qn")
    computed from this frame's own length distribution.

    rank(method="first") prevents qcut from failing when many documents have
    exactly the same effective length, especially after encoder truncation.
    """
    frame = frame.copy()

    if fixed_buckets is not None:
        categories = [label for label, _, _ in fixed_buckets]
        frame["bucket"] = pd.Categorical(
            frame["effective_token_length"].apply(
                lambda length: assign_fixed_length_bucket(length, fixed_buckets)
            ),
            categories=categories,
            ordered=True,
        )
        return frame

    ranks = frame["effective_token_length"].rank(method="first")

    bucket_codes = pd.qcut(
        ranks,
        q=num_buckets,
        labels=False,
        duplicates="drop",
    )
    bucket_numbers = bucket_codes.astype(int) + 1

    # Ordered Categorical so "Q10" sorts after "Q9", not before "Q2" as a plain string would.
    categories = [f"Q{n}" for n in range(1, num_buckets + 1)]
    frame["bucket"] = pd.Categorical(
        bucket_numbers.map(lambda n: f"Q{n}"), categories=categories, ordered=True
    )
    return frame


def p90(series: pd.Series) -> float:
    return float(series.quantile(0.90))


def summarize_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(
            "bucket",
            observed=True,
            sort=True,
        )
        .agg(
            documents=("document_id", "count"),
            min_effective_length=("effective_token_length", "min"),
            median_effective_length=("effective_token_length", "median"),
            max_effective_length=("effective_token_length", "max"),
            min_raw_length=("raw_token_length", "min"),
            median_raw_length=("raw_token_length", "median"),
            max_raw_length=("raw_token_length", "max"),
            truncation_rate=("was_truncated", "mean"),
            active_dims_mean=("active_dims", "mean"),
            active_dims_median=("active_dims", "median"),
            active_dims_p90=("active_dims", p90),
            sparsity_mean=("sparsity_ratio", "mean"),
            expansion_ratio_mean=("expansion_ratio", "mean"),
            expansion_weight_ratio_mean=("expansion_weight_ratio", "mean"),
            retention_ratio_mean=("retention_ratio", "mean"),
            retention_weight_ratio_mean=("retention_weight_ratio", "mean"),
        )
        .reset_index()
    )

    int_columns = [
        "min_effective_length",
        "max_effective_length",
        "min_raw_length",
        "max_raw_length",
    ]
    summary[int_columns] = summary[int_columns].astype(int)

    return summary.sort_values("bucket").reset_index(drop=True)


OVERALL_METRICS = (
    "active_dims",
    "sparsity_ratio",
    "expansion_ratio",
    "expansion_weight_ratio",
    "retention_ratio",
    "retention_weight_ratio",
)


def summarize_overall(frame: pd.DataFrame) -> pd.DataFrame:
    """Whole-dataset aggregate (mean/median/p90), independent of length bucket --
    the same view lm_toolkit/benchmarks/sparse_retrieval.py reports. Not
    derivable from summarize_buckets: medians/p90 don't combine across
    sub-groups the way means do."""
    records = [
        {
            "metric": metric,
            "mean": frame[metric].mean(),
            "median": frame[metric].median(),
            "p90": p90(frame[metric]),
        }
        for metric in OVERALL_METRICS
    ]
    return pd.DataFrame(records)


def calculate_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    length_columns = (
        "raw_token_length",
        "effective_token_length",
    )

    for length_column in length_columns:
        for metric in DOCUMENT_METRICS:
            valid = frame[[length_column, metric]].dropna()

            if len(valid) < 3:
                rho = math.nan
                p_value = math.nan
            else:
                result = spearmanr(valid[length_column], valid[metric])
                rho = float(result.statistic)
                p_value = float(result.pvalue)

            records.append(
                {
                    "length_variable": length_column,
                    "metric": metric,
                    "documents": len(valid),
                    "spearman_rho": rho,
                    "p_value": p_value,
                }
            )

    return pd.DataFrame(records)

def assign_fixed_length_bucket(
    effective_length: int,
    buckets: tuple[tuple[str, int, int | None], ...] = FIXED_LENGTH_BUCKETS,
) -> str:
    for label, low, high in buckets:
        if effective_length >= low and (high is None or effective_length <= high):
            return label
    raise ValueError(f"No fixed-length bucket matched length: {effective_length}")


def save_plots(
    frame: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    dataset = frame["dataset"].iloc[0]
    safe_name = dataset.lower().replace("/", "_").replace(" ", "_")

    # Hexbin avoids unreadable overplotting for thousands of documents.
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.hexbin(
        frame["effective_token_length"],
        frame["active_dims"],
        gridsize=45,
        mincnt=1,
    )
    ax.set_title(f"{dataset}: length vs active dimensions")
    ax.set_xlabel("Effective token length")
    ax.set_ylabel("Active dimensions")
    fig.colorbar(image, ax=ax, label="Documents per hexagon")
    fig.tight_layout()
    fig.savefig(
        output_dir / f"{safe_name}_length_vs_active_dims.png",
        dpi=180,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        bucket_summary["bucket"],
        bucket_summary["active_dims_mean"],
        marker="o",
        label="Mean",
    )
    ax.plot(
        bucket_summary["bucket"],
        bucket_summary["active_dims_median"],
        marker="o",
        label="Median",
    )
    ax.plot(
        bucket_summary["bucket"],
        bucket_summary["active_dims_p90"],
        marker="o",
        label="P90",
    )
    ax.set_title(f"{dataset}: active dimensions by length bucket")
    ax.set_xlabel("Length quantile")
    ax.set_ylabel("Active dimensions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / f"{safe_name}_bucket_active_dims.png",
        dpi=180,
    )
    plt.close(fig)


def print_key_results(
    overall_summary: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    print("\nOverall summary (whole dataset, not bucketed)")
    print(overall_summary.round(3).to_string(index=False))

    # Transposed: many metric columns but few buckets, so buckets-as-columns /
    # metrics-as-rows reads far better in a terminal than the wide CSV shape.
    print("\nBucket summary")
    print(bucket_summary.round(3).set_index("bucket").T.to_string())

    key_correlations = correlations[
        (correlations["length_variable"] == "effective_token_length")
        & (
            correlations["metric"].isin(
                [
                    "active_dims",
                    "sparsity_ratio",
                    "expansion_ratio",
                    "expansion_weight_ratio",
                ]
            )
        )
    ]

    print("\nKey Spearman correlations")
    print(key_correlations.to_string(index=False))

def task_hf_subset(task: RetrievalTask) -> str:
    if task.task_name == "CulturaViva-Retrieval":
        return "default"
    if task.task_name == "WebFAQRetrieval":
        return "ita"
    return task.language


def load_qrels(task: RetrievalTask) -> dict[str, set[str]]:
    lang_prefix = "ita" if task.task_name == "WebFAQRetrieval" else task.language

    if task.task_name == "CulturaViva-Retrieval":
        qrels_ds = load_dataset(
            path="lopozz/CulturaViva-Retrieval", name="qrels", split=task.split
        )
    else:
        qrels_ds = load_dataset(
            path=f"mteb/{task.task_name}", name=f"{lang_prefix}-qrels", split=task.split
        )

    relevant_docs: dict[str, set[str]] = defaultdict(set)
    for row in qrels_ds:
        if float(row["score"]) > 0:
            relevant_docs[str(row["query-id"])].add(str(row["corpus-id"]))
    return relevant_docs


def load_predictions(
    task: RetrievalTask,
    results_path: Path,
) -> dict[str, dict[str, float]] | None:
    """Read-only: never writes to results_dir. Produced separately by evaluate_mteb.py."""
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    hf_subset = task_hf_subset(task)

    return {
        str(query_id): {str(doc_id): float(score) for doc_id, score in documents.items()}
        for query_id, documents in payload[hf_subset][task.split].items()
    }


def ndcg_at_k(results: dict[str, float], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0

    ranked_ids = [
        doc_id for doc_id, _ in sorted(results.items(), key=lambda item: item[1], reverse=True)[:k]
    ]
    dcg = sum(
        1.0 / math.log2(rank + 2) for rank, doc_id in enumerate(ranked_ids) if doc_id in gold_ids
    )
    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def retrieval_by_length_bucket(
    document_frame: pd.DataFrame,
    task: RetrievalTask,
    results_path: Path,
    k: int = 10,
) -> pd.DataFrame:
    """
    Maps every query to the fixed-length bucket of its shortest gold document
    (the one most likely to suffer truncation effects first), computes
    per-query nDCG@k from saved evaluate_mteb.py predictions, and aggregates
    nDCG by bucket for this single task.

    Returns an empty DataFrame (caller should skip saving/printing) if no
    predictions file is found for this model/task -- this requires having
    already run evaluate_mteb.py.
    """
    predictions = load_predictions(task, results_path)

    relevant_docs = load_qrels(task)
    effective_length_by_doc = document_frame.set_index("document_id")["effective_token_length"].to_dict()

    records: list[dict[str, Any]] = []
    for query_id, gold_ids in relevant_docs.items():
        gold_lengths = [
            effective_length_by_doc[doc_id]
            for doc_id in gold_ids
            if doc_id in effective_length_by_doc
        ]
        if not gold_lengths:
            continue

        bucket = assign_fixed_length_bucket(min(gold_lengths))
        score = ndcg_at_k(predictions.get(query_id, {}), gold_ids, k)

        records.append(
            {
                "query_id": query_id,
                "fixed_length_bucket": bucket,
                "gold_doc_min_effective_length": min(gold_lengths),
                "num_gold_docs": len(gold_ids),
                f"ndcg_at_{k}": score,
            }
        )

    if not records:
        return pd.DataFrame()

    per_query = pd.DataFrame(records)
    bucket_order = [label for label, _, _ in FIXED_LENGTH_BUCKETS]
    per_query["fixed_length_bucket"] = pd.Categorical(
        per_query["fixed_length_bucket"], categories=bucket_order, ordered=True
    )

    return (
        per_query.groupby("fixed_length_bucket", observed=True)
        .agg(
            queries=("query_id", "count"),
            **{
                f"ndcg_at_{k}_mean": (f"ndcg_at_{k}", "mean"),
                f"ndcg_at_{k}_median": (f"ndcg_at_{k}", "median"),
            },
        )
        .reset_index()
        .sort_values("fixed_length_bucket")
        .reset_index(drop=True)
    )

def evaluate_sparse_retrieval(
    model: str,
    tasks: list[dict[str, Any]],
    backend: SparseEncoder,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    for task_config in tasks:
        if task_config["task_name"] not in DEFAULT_TASK_NAMES:
            raise ValueError(
                f"Unsupported task: {task_config['task_name']!r}. "
                f"Must be one of: {sorted(DEFAULT_TASK_NAMES)}"
            )

    backend.eval()

    num_buckets = 5
    fixed_buckets = None  # set to FIXED_LENGTH_BUCKETS to use fixed absolute-length ranges instead
    safe_model_name = model.replace("/", "__")

    results: list[dict[str, Any]] = []

    for task_config in tasks:
        task = RetrievalTask(**task_config)
        corpus = load_corpus(task)

        rows = encode_document_rows(
            task=task,
            corpus=corpus,
            backend=backend,
            batch_size=batch_size,
        )
        results.extend(rows)

        output_dir = Path("outputs") / "sparse_analysis"/ safe_model_name / task.task_name
        output_dir.mkdir(parents=True, exist_ok=True)

        document_frame = pd.DataFrame(rows)

        document_frame = add_dataset_quantile_buckets(
            document_frame,
            num_buckets=num_buckets,
            fixed_buckets=fixed_buckets,
        )
        overall_summary = summarize_overall(document_frame)
        bucket_summary = summarize_buckets(document_frame)
        correlations = calculate_correlations(document_frame)

        overall_summary.to_csv(
            output_dir / "overall_summary.csv",
            index=False,
        )
        bucket_summary.to_csv(
            output_dir / "bucket_summary.csv",
            index=False,
        )
        correlations.to_csv(
            output_dir / "spearman_correlations.csv",
            index=False,
        )

        results_path = Path("results") / safe_model_name / "prediction_folder" / f"{task.task_name}_predictions.json"

        if results_path.exists():
            retrieval_summary = retrieval_by_length_bucket(
                document_frame,
                task=task,
                results_path=results_path,
            )
            if not retrieval_summary.empty:
                retrieval_summary.to_csv(
                    output_dir / "retrieval_by_length_bucket.csv",
                    index=False,
                )
                print("\nRetrieval quality (nDCG@10) by gold-document length bucket")
                print(retrieval_summary.to_string(index=False))

        else:
            print(
                f"Skipping retrieval-by-length for {task.task_name}: no saved "
                f"predictions found for {model} under {results_path}. "
                "Run evaluate_mteb.py for this task/model first."
            )

        metadata = {
            "model": model,
            "task": task.task_name,
            "num_buckets": num_buckets,
            "documents": len(document_frame),
        }
        (output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        save_plots(
            document_frame,
            bucket_summary,
            output_dir,
        )
        print_key_results(
            overall_summary,
            bucket_summary,
            correlations,
        )

    return results