from __future__ import annotations

import math
import json
import torch

import numpy as np
import pandas as pd

from typing import Any
from pathlib import Path
from rich.progress import track
from datasets import load_dataset
from dataclasses import dataclass
from collections import defaultdict
from sentence_transformers import SparseEncoder

DOCUMENT_METRICS = (
    "active_dims",
    "vocab_size",
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

    if task.task_name == "CulturaViva-Retrieval":
        corpus_ds = load_dataset(
            path="lopozz/CulturaViva-Retrieval",
            name="corpus",
            split=task.split,
        )
    else:
        dataset_path = f"mteb/{task.task_name}"

        try:
            corpus_ds = load_dataset(
                path=dataset_path,
                name=f"{lang_prefix}-corpus",
                split=task.split,
            )
        except ValueError as error:
            # Fall back for datasets using plain configurations:
            # corpus, queries and qrels.
            if "BuilderConfig" not in str(error):
                raise

            corpus_ds = load_dataset(
                path=dataset_path,
                name="corpus",
                split=task.split,
            )

    id_column = "_id" if "_id" in corpus_ds.column_names else "id"

    return {
        str(row[id_column]): str(row["text"])
        for row in corpus_ds
    }


def load_queries(task: RetrievalTask) -> dict[str, str]:
    lang_prefix = (
        "ita" if task.task_name == "WebFAQRetrieval" else task.language
    )

    if task.task_name == "CulturaViva-Retrieval":
        queries_ds = load_dataset(
            path="lopozz/CulturaViva-Retrieval",
            name="queries",
            split=task.split,
        )
    else:
        dataset_path = f"mteb/{task.task_name}"

        try:
            queries_ds = load_dataset(
                path=dataset_path,
                name=f"{lang_prefix}-queries",
                split=task.split,
            )
        except ValueError as error:
            if "BuilderConfig" not in str(error):
                raise

            queries_ds = load_dataset(
                path=dataset_path,
                name="queries",
                split=task.split,
            )

    id_column = "_id" if "_id" in queries_ds.column_names else "id"

    return {
        str(row[id_column]): str(row["text"])
        for row in queries_ds
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
        "vocab_size": len(terms),                                                   # distinct token types in the effective input (length counts repeats, this doesn't)
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


def p10(series: pd.Series) -> float:
    return float(series.quantile(0.10))


def summarize_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(
            "bucket",
            observed=True,
            sort=True,
        )
        .agg(
            documents=("document_id", "count"),
            effective_length_p10=("effective_token_length", p10),
            effective_length_mean=("effective_token_length", "mean"),
            effective_length_median=("effective_token_length", "median"),
            effective_length_p90=("effective_token_length", p90),
            active_dims_p10=("active_dims", p10),
            active_dims_mean=("active_dims", "mean"),
            active_dims_median=("active_dims", "median"),
            active_dims_p90=("active_dims", p90),
            expansion_p10=("expansion", p10),
            expansion_mean=("expansion", "mean"),
            expansion_median=("expansion", "median"),
            expansion_p90=("expansion", p90),
            retention_p10=("retention", p10),
            retention_mean=("retention", "mean"),
            retention_median=("retention", "median"),
            retention_p90=("retention", p90),
            sparsity_mean=("sparsity_ratio", "mean"),
            expansion_ratio_mean=("expansion_ratio", "mean"),
            expansion_weight_ratio_mean=("expansion_weight_ratio", "mean"),
            retention_ratio_mean=("retention_ratio", "mean"),
            retention_weight_ratio_mean=("retention_weight_ratio", "mean"),
        )
        .reset_index()
    )

    return summary.sort_values("bucket").reset_index(drop=True)


OVERALL_METRICS = (
    "active_dims",
    "expansion",
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
    sub-groups the way means do.

    Covers only stats that require encoding text through the sparse encoder
    (active_dims, expansion, sparsity, ...). Token-length/vocab-size/
    truncation-rate stats don't need encoding at all -- see
    scripts/splade/analyze_dataset_stats.py for those, run independently as a
    cheap first look at a dataset before any encoder-dependent analysis."""
    records = [
        {
            "metric": metric,
            "p10": p10(frame[metric]),
            "mean": frame[metric].mean(),
            "median": frame[metric].median(),
            "p90": p90(frame[metric]),
        }
        for metric in OVERALL_METRICS
    ]
    return pd.DataFrame(records)


def assign_fixed_length_bucket(
    effective_length: int,
    buckets: tuple[tuple[str, int, int | None], ...] = FIXED_LENGTH_BUCKETS,
) -> str:
    for label, low, high in buckets:
        if effective_length >= low and (high is None or effective_length <= high):
            return label
    raise ValueError(f"No fixed-length bucket matched length: {effective_length}")


_STAT_SUFFIXES = ("p10", "mean", "median", "p90")


def split_metric_stat(name: str) -> tuple[str, str]:
    """Splits a bucket-summary column name like "active_dims_p90" into
    ("active_dims", "p90"), so the printed table can carry metric and stat
    as separate columns instead of one long joined name."""
    if name in ("documents", "queries"):
        return name, "count"

    metric, _, stat = name.rpartition("_")
    if metric and stat in _STAT_SUFFIXES:
        return metric, stat
    return name, ""


def transpose_by_stat(frame: pd.DataFrame, index_column: str) -> pd.DataFrame:
    """Transposes a bucket-summary-shaped frame (one row per bucket, one
    column per metric_stat) into metric/stat.-labeled rows with one column
    per bucket, blanking repeated metric names so each prints once above its
    stat rows."""
    transposed = frame.round(3).set_index(index_column).T
    metric_names, stat_names = zip(*(split_metric_stat(name) for name in transposed.index))
    deduped_metric_names = [
        name if index == 0 or name != metric_names[index - 1] else ""
        for index, name in enumerate(metric_names)
    ]
    transposed.insert(0, "stat.", stat_names)
    transposed.insert(0, "metric", deduped_metric_names)
    return transposed


def print_key_results(
    overall_summary: pd.DataFrame,
    retrieval_summary: pd.DataFrame,
    bucket_summary: pd.DataFrame,
) -> None:
    print("\nOverall summary")
    print(overall_summary.round(3).to_string(index=False))

    # Transposed: many metric columns but few buckets, so buckets-as-columns /
    # metrics-as-rows reads far better in a terminal than the wide CSV shape.
    if retrieval_summary is not None:
        print("\nnDCG@10 by bucket")
        print(transpose_by_stat(retrieval_summary, "bucket").to_string(index=False))

    print("\nBucket summary")
    print(transpose_by_stat(bucket_summary, "bucket").to_string(index=False))


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
    Maps every query to the bucket of its shortest gold document (the one
    most likely to suffer truncation effects first), reusing document_frame's
    own "bucket" assignment -- so this always matches whichever scheme
    (quantile or fixed) was used for the rest of the run, instead of
    recomputing bucket edges independently. Computes per-query nDCG@k from
    saved evaluate_mteb.py predictions, and aggregates nDCG by bucket.

    Returns an empty DataFrame (caller should skip saving/printing) if no
    predictions file is found for this model/task -- this requires having
    already run evaluate_mteb.py.
    """
    predictions = load_predictions(task, results_path)

    relevant_docs = load_qrels(task)
    bucket_by_doc = document_frame.set_index("document_id")["bucket"]

    records: list[dict[str, Any]] = []
    for query_id, gold_ids in relevant_docs.items():
        present_ids = [doc_id for doc_id in gold_ids if doc_id in bucket_by_doc.index]
        if not present_ids:
            continue

        bucket = bucket_by_doc.loc[present_ids].min()  # respects the ordered Categorical
        score = ndcg_at_k(predictions.get(query_id, {}), gold_ids, k)

        records.append(
            {
                "query_id": query_id,
                "bucket": bucket,
                "num_gold_docs": len(gold_ids),
                f"ndcg_at_{k}": score,
            }
        )

    if not records:
        return pd.DataFrame()

    per_query = pd.DataFrame(records)
    per_query["bucket"] = pd.Categorical(
        per_query["bucket"], categories=bucket_by_doc.cat.categories, ordered=True
    )

    return (
        per_query.groupby("bucket", observed=True)
        .agg(
            queries=("query_id", "count"),
            **{
                f"ndcg_at_{k}_mean": (f"ndcg_at_{k}", "mean"),
                f"ndcg_at_{k}_median": (f"ndcg_at_{k}", "median"),
            },
        )
        .reset_index()
        .sort_values("bucket")
        .reset_index(drop=True)
    )


def build_fp_audit(
    relevant_docs: dict[str, set[str]],
    predictions: dict[str, dict[str, float]],
    active_dims_by_doc: pd.Series,
    length_by_doc: pd.Series,
    bucket_by_doc: pd.Series,
    k: int = 10,
) -> pd.DataFrame:
    """
    For every query with NDCG@k < 1.0, finds the gold document the model
    itself scored highest (its "best shot", for queries with more than one
    gold doc) and the top-ranked false positive, then reports both documents'
    score/active_dims/length/bucket plus the score_gap between them -- for
    auditing whether retrieval failures come from the true doc scoring low,
    the false positive scoring anomalously high, or both.
    """
    records: list[dict[str, Any]] = []

    for query_id, gold_ids in relevant_docs.items():
        results = predictions.get(query_id, {})
        if not results:
            continue

        score = ndcg_at_k(results, gold_ids, k)
        if score >= 1.0:
            continue  # only auditing imperfect queries

        ranked_ids = [
            doc_id for doc_id, _ in sorted(results.items(), key=lambda item: item[1], reverse=True)
        ]

        gold_present = [doc_id for doc_id in gold_ids if doc_id in results]
        if not gold_present:
            continue

        true_doc_id = max(gold_present, key=lambda doc_id: results[doc_id])
        true_doc_score = results[true_doc_id]
        true_doc_rank = ranked_ids.index(true_doc_id) + 1

        top_fp_id = next(doc_id for doc_id in ranked_ids if doc_id not in gold_ids)
        top_fp_score = results[top_fp_id]

        records.append(
            {
                "query_id": query_id,
                f"ndcg_at_{k}": score,
                "true_doc_id": true_doc_id,
                "true_doc_score": true_doc_score,
                "true_doc_rank": true_doc_rank,
                "true_doc_active_dims": active_dims_by_doc.get(true_doc_id),
                "true_doc_length": length_by_doc.get(true_doc_id),
                "true_doc_bucket": bucket_by_doc.get(true_doc_id),
                "top_fp_id": top_fp_id,
                "top_fp_score": top_fp_score,
                "top_fp_active_dims": active_dims_by_doc.get(top_fp_id),
                "top_fp_length": length_by_doc.get(top_fp_id),
                "top_fp_bucket": bucket_by_doc.get(top_fp_id),
                "score_gap": top_fp_score - true_doc_score,
            }
        )

    return pd.DataFrame(records)


def add_audit_flags(audit: pd.DataFrame) -> pd.DataFrame:
    """
    Adds H1 (short + dense false positives beat the true doc) comparison
    columns to a build_fp_audit table: booleans for whether the false
    positive actually outranks/outscores the true doc and whether it's
    shorter/denser, plus the same two comparisons as continuous differences
    and ratios instead of just yes/no.
    """
    audit = audit.copy()

    audit["fp_outranks_true"] = audit["top_fp_score"] > audit["true_doc_score"]
    audit["top_result_is_fp"] = audit["true_doc_rank"] > 1

    audit["fp_shorter"] = audit["top_fp_length"] < audit["true_doc_length"]
    audit["fp_more_active"] = audit["top_fp_active_dims"] > audit["true_doc_active_dims"]
    audit["fp_shorter_and_more_active"] = audit["fp_shorter"] & audit["fp_more_active"]

    audit["length_difference"] = audit["top_fp_length"] - audit["true_doc_length"]
    audit["active_dims_difference"] = audit["top_fp_active_dims"] - audit["true_doc_active_dims"]

    audit["active_dims_ratio"] = (
        audit["top_fp_active_dims"] / audit["true_doc_active_dims"].replace(0, np.nan)
    )
    audit["length_ratio"] = audit["top_fp_length"] / audit["true_doc_length"].replace(0, np.nan)

    return audit


def summarize_displacement(audit: pd.DataFrame, displacement: pd.DataFrame) -> pd.Series:
    """
    displacement is the subset of an add_audit_flags-flagged audit table
    where fp_outranks_true is True -- queries where a genuinely wrong
    document outscored the true one. Summarizes H1 (short + dense false
    positives beat the true doc) as a single set of headline numbers.
    """
    return pd.Series(
        {
            "audited_queries": len(audit),
            "displacement_queries": len(displacement),
            "median_true_length": displacement["true_doc_length"].median(),
            "median_fp_length": displacement["top_fp_length"].median(),
            "median_true_active_dims": displacement["true_doc_active_dims"].median(),
            "median_fp_active_dims": displacement["top_fp_active_dims"].median(),
            "share_fp_shorter": displacement["fp_shorter"].mean(),
            "share_fp_more_active": displacement["fp_more_active"].mean(),
            "share_fp_shorter_and_more_active": displacement["fp_shorter_and_more_active"].mean(),
            "median_score_gap": displacement["score_gap"].median(),
            "median_length_difference": displacement["length_difference"].median(),
            "median_active_dims_difference": displacement["active_dims_difference"].median(),
        }
    )


def summarize_displacement_by_bucket(displacement: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Same H1 breakdown as summarize_displacement, grouped by the true doc's
    own length bucket -- so you can see whether displacement concentrates in
    a specific length range rather than being spread evenly."""
    return (
        displacement.groupby("true_doc_bucket", observed=True)
        .agg(
            queries=("query_id", "size"),
            ndcg_mean=(f"ndcg_at_{k}", "mean"),
            true_rank_median=("true_doc_rank", "median"),
            score_gap_mean=("score_gap", "mean"),
            score_gap_median=("score_gap", "median"),
            true_length_median=("true_doc_length", "median"),
            fp_length_median=("top_fp_length", "median"),
            true_active_dims_median=("true_doc_active_dims", "median"),
            fp_active_dims_median=("top_fp_active_dims", "median"),
            share_fp_shorter=("fp_shorter", "mean"),
            share_fp_more_active=("fp_more_active", "mean"),
            share_fp_shorter_and_more_active=("fp_shorter_and_more_active", "mean"),
        )
        .reset_index()
    )


def compute_bucket_enrichment(document_frame: pd.DataFrame, displacement: pd.DataFrame) -> pd.DataFrame:
    """
    Compares each length bucket's share of the whole corpus against its share
    of top false positives. An enrichment_ratio above 1 means that bucket is
    overrepresented among false positives relative to how common it actually
    is in the corpus -- stronger evidence for H1 than a raw share_fp_shorter
    number, since it controls for how common short documents are to begin with.
    """
    corpus_share = document_frame["bucket"].value_counts(normalize=True).sort_index()
    fp_share = displacement["top_fp_bucket"].value_counts(normalize=True).sort_index()

    enrichment = pd.concat(
        [corpus_share.rename("corpus_share"), fp_share.rename("false_positive_share")],
        axis=1,
    )
    enrichment["false_positive_share"] = enrichment["false_positive_share"].fillna(0.0)
    enrichment["enrichment_ratio"] = (
        enrichment["false_positive_share"] / enrichment["corpus_share"].replace(0, np.nan)
    )
    return enrichment.reset_index(names="bucket")


def compute_fp_active_dims_baseline(document_frame: pd.DataFrame, displacement: pd.DataFrame) -> pd.DataFrame:
    """
    Adds fp_active_excess and fp_active_zscore to displacement: how far the
    false positive's active_dims sits above its OWN bucket's median/mean --
    i.e. is it anomalously dense even for documents of its own length, not
    just denser than the (differently-bucketed) true doc.
    """
    bucket_active_baseline = (
        document_frame.groupby("bucket", observed=True)["active_dims"]
        .agg(["mean", "median", "std"])
        .rename(
            columns={
                "mean": "bucket_active_mean",
                "median": "bucket_active_median",
                "std": "bucket_active_std",
            }
        )
    )

    displacement = displacement.join(bucket_active_baseline, on="top_fp_bucket")

    displacement["fp_active_excess"] = (
        displacement["top_fp_active_dims"] - displacement["bucket_active_median"]
    )
    displacement["fp_active_zscore"] = (
        displacement["top_fp_active_dims"] - displacement["bucket_active_mean"]
    ) / displacement["bucket_active_std"].replace(0, np.nan)

    return displacement


def run_fp_audit(
    document_frame: pd.DataFrame,
    relevant_docs: dict[str, set[str]],
    predictions: dict[str, dict[str, float]],
    k: int = 10,
) -> dict[str, Any]:
    """
    Runs the full H1 (short + dense false positives beat the true doc) audit
    in one call: builds the per-query audit, flags the H1 comparisons, and
    computes all four summaries. Everything here is cheap pandas work over
    already-loaded predictions/qrels/document_frame -- nothing gets
    re-encoded, so this is safe to call repeatedly from a notebook.
    """
    active_dims_by_doc = document_frame.set_index("document_id")["active_dims"]
    length_by_doc = document_frame.set_index("document_id")["effective_token_length"]
    bucket_by_doc = document_frame.set_index("document_id")["bucket"]

    audit = add_audit_flags(
        build_fp_audit(relevant_docs, predictions, active_dims_by_doc, length_by_doc, bucket_by_doc, k=k)
    )
    displacement = audit[audit["fp_outranks_true"]].copy()

    return {
        "audit": audit,
        "displacement": displacement,
        "displacement_summary": summarize_displacement(audit, displacement),
        "displacement_by_bucket": summarize_displacement_by_bucket(displacement, k=k),
        "bucket_enrichment": compute_bucket_enrichment(document_frame, displacement),
        "fp_active_baseline": compute_fp_active_dims_baseline(document_frame, displacement),
    }


def print_fp_audit_results(results: dict[str, Any]) -> None:
    """
    H1 audit: among queries the model got wrong, does the winning false
    positive tend to be shorter and denser (more active dims) than the true
    (gold) document? Each section below narrows the question:
      1. does it happen, and how often, overall
      2. where it concentrates (by the true doc's own length bucket)
      3. whether short/dense documents are overrepresented among false
         positives relative to their actual share of the corpus
      4. whether the winning false positive is anomalously dense even
         relative to other documents of its own length (not just denser
         than the differently-sized true doc it beat)
    """
    print("\nDisplacement summary: does H1 happen, and how often, overall?")
    print(results["displacement_summary"].round(2).to_string())

    print(
        "\nDisplacement by bucket: does it concentrate at a particular true-doc "
        "length? (queries = number of displaced queries whose true doc falls in "
        "that bucket; share_fp_* = fraction of those where the false positive "
        "was shorter / denser / both)"
    )
    print(results["displacement_by_bucket"].round(2).set_index("true_doc_bucket").T.to_string())

    print(
        "\nBucket enrichment: are short/dense buckets overrepresented among false "
        "positives relative to how common they actually are in the corpus? "
        "(enrichment_ratio > 1 = overrepresented, < 1 = underrepresented)"
    )
    print(results["bucket_enrichment"].round(2).to_string(index=False))

    print(
        "\nFP active-dims anomaly: is the winning false positive denser than even "
        "other documents of its OWN length (not just denser than the true doc it "
        "beat)? fp_active_excess = raw gap over its bucket's median active_dims; "
        "fp_active_zscore = same gap in standard-deviation units."
    )
    baseline = results["fp_active_baseline"]
    print(
        baseline[["fp_active_excess", "fp_active_zscore"]]
        .describe()
        .round(2)
        .to_string()
    )
    print()
    print(
        baseline.groupby("true_doc_bucket", observed=True)
        .agg(
            fp_active_excess_median=("fp_active_excess", "median"),
            fp_active_zscore_mean=("fp_active_zscore", "mean"),
            share_fp_above_bucket_median=("fp_active_excess", lambda values: (values > 0).mean()),
        )
        .round(2)
        .to_string()
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

        # Per-document rows (active_dims, effective_token_length, bucket, ...),
        # so downstream analyses (e.g. a per-query error audit) can look up
        # document-level metrics without re-encoding the whole corpus.
        document_frame.to_csv(
            output_dir / "document_frame.csv",
            index=False,
        )
        overall_summary.to_csv(
            output_dir / "overall_summary.csv",
            index=False,
        )
        bucket_summary.to_csv(
            output_dir / "bucket_summary.csv",
            index=False,
        )

        results_path = Path("results") / safe_model_name / "prediction_folder" / f"{task.task_name}_predictions.json"

        if results_path.exists():
            retrieval_summary = retrieval_by_length_bucket(
                document_frame,
                task=task,
                results_path=results_path,
            )

            assert not retrieval_summary.empty, f"Expected non-empty retrieval summary for {task.task_name} at {results_path}"

            retrieval_summary.to_csv(
                output_dir / "retrieval_by_length_bucket.csv",
                index=False,
            )

            # Reuses the same predictions/qrels already needed above -- cheap,
            # since document_frame (the expensive part) is already in memory.
            predictions = load_predictions(task, results_path)
            relevant_docs = load_qrels(task)
            fp_audit_results = run_fp_audit(document_frame, relevant_docs, predictions, k=10)
        else:
            retrieval_summary = None
            print(
                f"Skipping retrieval-by-length for {task.task_name}: no saved "
                f"predictions found for {model} under {results_path}. "
                "Run evaluate_mteb.py for this task/model first."
            )

        print_key_results(overall_summary, retrieval_summary, bucket_summary)
        if results_path.exists(): 
            print_fp_audit_results(fp_audit_results)

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

    return results