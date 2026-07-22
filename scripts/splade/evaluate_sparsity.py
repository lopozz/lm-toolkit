"""
Evaluate sparse-encoder activation behavior over a full retrieval dataset.

Unlike evaluate_mteb.py, which measures ranking quality (nDCG, MAP, recall),
this script measures how a SPLADE-style model represents queries and documents:

- NNZ: Number of vocabulary dimensions with non-zero activation.
- Expansion ratio: Activated dimensions relative to the number of input tokens.
- Expansion weight mass: Total activation weight assigned to terms not present
  in the original input.
- Lexical retention: Proportion of original input terms retained with non-zero
  activation.

Metrics are aggregated using mean, median, and 90th percentile over every query
and every document in the corpus. Run the script once per model to compare
representation behavior across models on the same dataset.

Typical usage:

    python3 scripts/splade/evaluate_sparsity.py \\
      --model nickprock/splade-bert-base-italian-xxl-uncased-cv \\
      --task-name MuPLeR-retrieval --language it
"""

import argparse
import lm_toolkit

from sentence_transformers import SparseEncoder
from sentence_transformers.sentence_transformer.modules import Router, Transformer
from sentence_transformers.sparse_encoder.modules import SparseStaticEmbedding, SpladePooling


def load_backend(model_name: str, device: str, query_expansion: bool) -> SparseEncoder:
    if query_expansion:
        return SparseEncoder(model_name, device=device)

    # Force the query side through a SparseStaticEmbedding (lexical reweighting
    # of input tokens only, no MLM expansion), while documents keep the full
    # Transformer + SpladePooling route. Mirrors the router shape already used
    # by the local asymmetric SPLADE checkpoints in this project.
    mlm_transformer = Transformer(model_name, transformer_task="fill-mask")
    splade_pooling = SpladePooling(
        pooling_strategy="max",
        embedding_dimension=mlm_transformer.get_embedding_dimension(),
    )
    router = Router.for_query_document(
        query_modules=[
            SparseStaticEmbedding(tokenizer=mlm_transformer.tokenizer, frozen=True)
        ],
        document_modules=[mlm_transformer, splade_pooling],
    )
    return SparseEncoder(modules=[router], device=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SPLADE sparse-activation behavior over a full retrieval dataset."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="SentenceTransformers SparseEncoder model name or local path.",
    )
    parser.add_argument("--task-name", default="MuPLeR-retrieval")
    parser.add_argument("--language", default="it")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--query-limit",
        type=int,
        help="Optional cap on number of queries encoded (default: all).",
    )
    parser.add_argument(
        "--document-limit",
        type=int,
        help="Optional cap on number of documents encoded (default: all).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-query-expansion",
        action="store_false",
        dest="query_expansion",
        help="Replace the query route with a SparseStaticEmbedding "
        "(lexical reweighting only, no MLM expansion), for a like-for-like "
        "comparison against models that don't expand queries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    backend = load_backend(args.model, args.device, args.query_expansion)

    lm_toolkit.evaluate(
        model=args.model,
        tasks=[
            {
                "task_name": args.task_name,
                "language": args.language,
                "split": args.split,
            }
        ],
        backend=backend,
        benchmark="sparse_retrieval",
        kwargs={
            "batch_size": args.batch_size,
        },
    )


if __name__ == "__main__":
    main()
