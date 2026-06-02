from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import bm25s
import torch
from datasets import load_dataset
from rich.progress import track
from sentence_transformers import SparseEncoder


DEFAULT_MODELS = {
    "opensearch_sparse": "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
    "splade_it": "nickprock/splade-bert-base-italian-xxl-uncased-cv",
}
DEFAULT_BM25_LABEL = "bm25"
DEFAULT_BM25_MODEL = "mteb/baseline-bm25s"
DISPLAY_DOCUMENTS = 5
NDCG_K = 10

Expansion = list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the JSON consumed by sparse_retrieval_comparison_app.py.",
    )
    parser.add_argument("--task-name", default="MuPLeR-retrieval")
    parser.add_argument("--language", default="it")
    parser.add_argument("--split", default="test")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("streamlits/sparse_retrieval_comparison.json"),
    )
    parser.add_argument(
        "--top-terms",
        type=int,
        default=100,
        help="Maximum number of active terms stored per representation. Use -1 to keep all terms.",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        help="Optional limit for faster exploratory exports.",
    )
    parser.add_argument(
        "--query-id",
        action="append",
        dest="query_ids",
        help="Export only this query ID. May be repeated.",
    )
    parser.add_argument(
        "--splade-model",
        action="append",
        metavar="LABEL=MODEL_NAME",
        help="SPLADE model to export. May be repeated. Defaults to the two project models.",
    )
    parser.add_argument("--bm25-label", default=DEFAULT_BM25_LABEL)
    parser.add_argument("--bm25-model-name", default=DEFAULT_BM25_MODEL)
    parser.add_argument("--skip-bm25", action="store_true")
    return parser.parse_args()


def normalize_model_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def prediction_path(results_dir: Path, model_name: str, task_name: str) -> Path:
    return (
        results_dir
        / normalize_model_name(model_name)
        / "prediction_folder"
        / f"{task_name}_predictions.json"
    )


def load_predictions(
    results_dir: Path,
    model_name: str,
    task_name: str,
    language: str,
    split: str,
) -> dict[str, dict[str, float]]:
    path = prediction_path(results_dir, model_name, task_name)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(query_id): {str(doc_id): float(score) for doc_id, score in documents.items()}
        for query_id, documents in payload[language][split].items()
    }


def load_texts(
    task_name: str,
    language: str,
    split: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    params = {"path": f"mteb/{task_name}", "split": split}
    corpus_ds = load_dataset(name=f"{language}-corpus", **params)
    queries_ds = load_dataset(name=f"{language}-queries", **params)
    qrels_ds = load_dataset(name=f"{language}-qrels", **params)
    corpus = {str(row["id"]): str(row["text"]) for row in corpus_ds}
    queries = {str(row["id"]): str(row["text"]) for row in queries_ds}
    gold_ids: dict[str, set[str]] = defaultdict(set)
    for row in qrels_ds:
        if float(row["score"]) > 0:
            gold_ids[str(row["query-id"])].add(str(row["corpus-id"]))
    return corpus, queries, gold_ids


def parse_models(values: list[str] | None) -> dict[str, str]:
    if not values:
        return dict(DEFAULT_MODELS)

    models = {}
    for value in values:
        try:
            label, model_name = value.split("=", maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"Expected LABEL=MODEL_NAME, received: {value}") from exc
        if not label or not model_name:
            raise ValueError(f"Expected non-empty LABEL=MODEL_NAME, received: {value}")
        if label in models:
            raise ValueError(f"Duplicate SPLADE model label: {label}")
        models[label] = model_name
    return models


def validate_args(args: argparse.Namespace, splade_models: dict[str, str]) -> None:
    if args.top_terms != -1 and args.top_terms < 1:
        raise ValueError("--top-terms must be -1 or a positive integer.")
    if args.query_limit is not None and args.query_limit < 1:
        raise ValueError("--query-limit must be a positive integer.")
    if not args.skip_bm25 and args.bm25_label in splade_models:
        raise ValueError(f"BM25 label collides with a SPLADE model label: {args.bm25_label}")


def ranked_documents(results: dict[str, float], top_documents: int) -> list[tuple[str, float]]:
    return sorted(results.items(), key=lambda item: item[1], reverse=True)[:top_documents]


def displayed_documents(results: dict[str, float], gold_ids: set[str]) -> list[dict[str, Any]]:
    ranked = ranked_documents(results, len(results))
    ranked_lookup = {
        doc_id: {"rank": rank, "score": score}
        for rank, (doc_id, score) in enumerate(ranked, start=1)
    }
    top_documents = [
        {
            "doc_id": doc_id,
            "rank": rank,
            "score": score,
            "is_extra_gold": False,
        }
        for rank, (doc_id, score) in enumerate(ranked[:DISPLAY_DOCUMENTS], start=1)
    ]
    top_ids = {document["doc_id"] for document in top_documents}
    missing_gold = sorted(
        gold_ids - top_ids,
        key=lambda doc_id: (ranked_lookup.get(doc_id, {}).get("rank", math.inf), doc_id),
    )
    extra_gold = [
        {
            "doc_id": doc_id,
            "rank": ranked_lookup.get(doc_id, {}).get("rank"),
            "score": ranked_lookup.get(doc_id, {}).get("score"),
            "is_extra_gold": True,
        }
        for doc_id in missing_gold
    ]
    return extra_gold + top_documents


def ndcg_at_k(results: dict[str, float], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0

    ranked_ids = [doc_id for doc_id, _ in ranked_documents(results, k)]
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, doc_id in enumerate(ranked_ids)
        if doc_id in gold_ids
    )
    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def limit_expansion(expansion: Expansion, top_terms: int) -> Expansion:
    expansion.sort(key=lambda item: (-float(item["weight"]), str(item["token"])))
    return expansion if top_terms == -1 else expansion[:top_terms]


def tensor_expansion(vector: Any, vocab: dict[int, str], top_terms: int) -> Expansion:
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
    return limit_expansion(expansion, top_terms)


def encode_splade_texts(
    texts: dict[str, str],
    model_name: str,
    mode: str,
    top_terms: int,
    batch_size: int = 16,
) -> dict[str, Expansion]:
    if not texts:
        return {}

    model = SparseEncoder(model_name, device="cuda")
    model.eval()
    vocab = {index: token for token, index in model.tokenizer.get_vocab().items()}
    text_ids = list(texts)
    encoded: dict[str, Expansion] = {}

    for start in track(
        range(0, len(text_ids), batch_size),
        description=f"[SPLADE {mode}] {model_name}",
    ):
        batch_ids = text_ids[start : start + batch_size]
        batch_texts = [texts[text_id] for text_id in batch_ids]
        with torch.no_grad():
            if mode == "query":
                vectors = model.encode_query(batch_texts, convert_to_sparse_tensor=False)
            else:
                vectors = model.encode_document(batch_texts, convert_to_sparse_tensor=False)

        for text_id, vector in zip(batch_ids, vectors, strict=True):
            encoded[text_id] = tensor_expansion(vector, vocab, top_terms)

    del model
    return encoded


def build_bm25_index(corpus: dict[str, str]) -> tuple[bm25s.BM25, list[str]]:
    corpus_ids = list(corpus)
    corpus_tokens = bm25s.tokenize([corpus[doc_id] for doc_id in corpus_ids])
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    return retriever, corpus_ids


def bm25_query_expansion(text: str, retriever: bm25s.BM25, top_terms: int) -> Expansion:
    tokens = bm25s.tokenize([text], return_ids=False, show_progress=False)[0]
    counts = Counter(token for token in tokens if token in retriever.vocab_dict)
    return limit_expansion(
        [{"token": token, "weight": float(weight)} for token, weight in counts.items()],
        top_terms,
    )


def bm25_document_expansions(
    retriever: bm25s.BM25,
    corpus_ids: list[str],
    selected_doc_ids: set[str],
    top_terms: int,
) -> dict[str, Expansion]:
    selected_positions = {
        position: doc_id
        for position, doc_id in enumerate(corpus_ids)
        if doc_id in selected_doc_ids
    }
    expansions: dict[str, Expansion] = defaultdict(list)
    inverse_vocab = {token_id: token for token, token_id in retriever.vocab_dict.items()}
    data = retriever.scores["data"]
    indices = retriever.scores["indices"]
    indptr = retriever.scores["indptr"]

    indexed_tokens = (
        (token_id, token)
        for token_id, token in inverse_vocab.items()
        if token_id + 1 < len(indptr)
    )
    for token_id, token in track(indexed_tokens, description="[BM25] Collecting document weights"):
        for offset in range(indptr[token_id], indptr[token_id + 1]):
            doc_id = selected_positions.get(int(indices[offset]))
            if doc_id is not None and float(data[offset]) > 0:
                expansions[doc_id].append({"token": token, "weight": float(data[offset])})

    return {
        doc_id: limit_expansion(expansions.get(doc_id, []), top_terms)
        for doc_id in selected_doc_ids
    }


def selected_query_ids(
    predictions: dict[str, dict[str, dict[str, float]]],
    query_ids: list[str] | None,
    query_limit: int | None,
) -> list[str]:
    common_ids = set.intersection(*(set(model_predictions) for model_predictions in predictions.values()))
    if query_ids:
        requested_ids = [str(query_id) for query_id in query_ids]
        missing_ids = [query_id for query_id in requested_ids if query_id not in common_ids]
        if missing_ids:
            raise ValueError(f"Query IDs missing from at least one prediction file: {missing_ids}")
        return requested_ids[:query_limit]

    return sorted(common_ids)[:query_limit]


def generate_comparison(args: argparse.Namespace) -> dict[str, Any]:
    splade_models = parse_models(args.splade_model)
    validate_args(args, splade_models)
    model_names = dict(splade_models)
    if not args.skip_bm25:
        model_names[args.bm25_label] = args.bm25_model_name

    corpus, queries, gold_ids = load_texts(args.task_name, args.language, args.split)
    predictions = {
        label: load_predictions(
            args.results_dir,
            model_name,
            args.task_name,
            args.language,
            args.split,
        )
        for label, model_name in model_names.items()
    }
    query_ids = selected_query_ids(predictions, args.query_ids, args.query_limit)
    missing_queries = [query_id for query_id in query_ids if query_id not in queries]
    if missing_queries:
        raise ValueError(f"Query IDs missing from the dataset: {missing_queries}")

    displayed_by_model = {
        label: {
            query_id: displayed_documents(model_predictions[query_id], gold_ids.get(query_id, set()))
            for query_id in query_ids
        }
        for label, model_predictions in predictions.items()
    }
    selected_docs_by_model = {
        label: {
            document["doc_id"]
            for displayed_documents_for_query in displayed_queries.values()
            for document in displayed_documents_for_query
        }
        for label, displayed_queries in displayed_by_model.items()
    }
    missing_docs = sorted(
        {
            doc_id
            for model_docs in selected_docs_by_model.values()
            for doc_id in model_docs
            if doc_id not in corpus
        }
    )
    if missing_docs:
        raise ValueError(f"Retrieved document IDs missing from the dataset: {missing_docs[:10]}")

    query_representations: dict[str, dict[str, Expansion]] = {}
    document_expansions: dict[str, dict[str, Expansion]] = {}
    selected_query_texts = {query_id: queries[query_id] for query_id in query_ids}

    for label, model_name in splade_models.items():
        selected_document_texts = {
            doc_id: corpus[doc_id] for doc_id in selected_docs_by_model[label]
        }
        document_expansions[label] = encode_splade_texts(
            selected_document_texts,
            model_name,
            mode="doc",
            top_terms=args.top_terms,
        )
        query_representations[label] = encode_splade_texts(
            selected_query_texts,
            model_name,
            mode="query",
            top_terms=args.top_terms,
        )

    if not args.skip_bm25:
        retriever, corpus_ids = build_bm25_index(corpus)
        document_expansions[args.bm25_label] = bm25_document_expansions(
            retriever,
            corpus_ids,
            selected_docs_by_model[args.bm25_label],
            args.top_terms,
        )
        query_representations[args.bm25_label] = {
            query_id: bm25_query_expansion(queries[query_id], retriever, args.top_terms)
            for query_id in query_ids
        }

    output_queries = []
    for query_id in query_ids:
        models = {}
        for label in model_names:
            documents = [
                {
                    **document,
                    "text": corpus[document["doc_id"]],
                    "expansion": document_expansions[label][document["doc_id"]],
                }
                for document in displayed_by_model[label][query_id]
            ]
            models[label] = {
                "ndcg_at_10": ndcg_at_k(predictions[label][query_id], gold_ids.get(query_id, set()), NDCG_K),
                "query_representation": query_representations[label][query_id],
                "documents": documents,
            }

        output_queries.append(
            {
                "qid": query_id,
                "text": queries[query_id],
                "gold_ids": sorted(gold_ids.get(query_id, set())),
                "models": models,
            }
        )

    return {"queries": output_queries}


def main() -> None:
    args = parse_args()
    payload = generate_comparison(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved comparison data to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
