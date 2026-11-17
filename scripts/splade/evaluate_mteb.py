r"""
Evaluate dense, BM25 and sparse (SPLADE-style) retrieval models with native MTEB.

Model loading:
    - `--inference-free` builds a custom `SparseEncoder` (static query route)
      and wraps it in MTEB's `SparseEncoderWrapper`;
    - everything else uses `mteb.get_model(model_name)`, which auto-detects
      dense, BM25 and sparse models.

`--max-active-dims` is forwarded to `SparseEncoder.encode` through MTEB's
`encode_kwargs`, so it prunes both queries and documents.

Examples
--------
BM25 baseline:
    uv run python scripts/splade/evaluate_mteb.py \
        --model mteb/baseline-bm25s

SPLADE model:
    uv run python scripts/splade/evaluate_mteb.py \
        --model lopozz/splade-bert-base-italian-xxl-uncased

SPLADE model, inference-free queries and top-64 pruning:
    uv run python scripts/splade/evaluate_mteb.py \
        --model nickprock/splade-bert-base-italian-xxl-uncased-cv \
        --inference-free

Dense model:
    uv run python scripts/splade/evaluate_mteb.py \
        --model google/embeddinggemma-300m --batch-size 32
"""

import argparse
from pathlib import Path

import mteb
import torch
import yaml
from huggingface_hub import file_exists
from mteb.models.sentence_transformer_wrapper import SparseEncoderWrapper
from sentence_transformers import SparseEncoder
from sentence_transformers.sentence_transformer.modules import Router, Transformer
from sentence_transformers.sparse_encoder.modules import (
    SparseStaticEmbedding,
    SpladePooling,
)

from lm_toolkit.tasks.retrieval.ita import CulturaVivaRetrieval, MMarcoITRetrieval
from lm_toolkit.tasks.retrieval.multilingual import (
    GermanDPRRetrieval,
    JaQuADRetrieval,
    NorQuADRetrieval,
    SberQuADRetrieval,
)


def has_router(model_name: str) -> bool:
    """True if the checkpoint (local dir or Hub repo) was saved with a Router."""
    if Path(model_name).is_dir():
        return (Path(model_name) / "router_config.json").exists()
    try:
        return file_exists(model_name, "router_config.json")
    except Exception:
        return False


def load_sparse_model(model_name: str, inference_free: bool) -> SparseEncoder:
    if not inference_free:
        return SparseEncoder(model_name)

    if has_router(model_name):
        raise ValueError(
            f"{model_name} already has a query/document Router (it was saved with "
            "its own query route, e.g. trained inference-free), so --inference-free "
            "cannot be applied. Run without --inference-free to use it as saved."
        )

    # Query side: SparseStaticEmbedding (lexical reweighting only, no MLM
    # expansion). Document side: full Transformer + SpladePooling.
    mlm_transformer = Transformer(model_name, transformer_task="fill-mask")
    splade_pooling = SpladePooling(
        pooling_strategy="max",
        embedding_dimension=mlm_transformer.get_embedding_dimension(),
    )
    # MLM head width is config.vocab_size, which can exceed len(tokenizer vocab)
    # (e.g. dbmdz/bert-base-italian-xxl-uncased); match it so query and document
    # vectors share the same dimensionality.
    vocab_size = mlm_transformer.model.config.vocab_size
    router = Router.for_query_document(
        query_modules=[
            SparseStaticEmbedding(
                tokenizer=mlm_transformer.tokenizer,
                weight=torch.ones(vocab_size),
                frozen=True,
            )
        ],
        document_modules=[mlm_transformer, splade_pooling],
    )
    return SparseEncoder(modules=[router])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval models with native MTEB (dense, BM25, sparse)."
    )
    parser.add_argument("--model", required=True, help="Model name or HF repo.")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("./configs/ds/mteb-retrieval-ita.yml"),
        help="Dataset task configuration YAML.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite-strategy", default="only-missing")
    parser.add_argument("--results-dir", type=Path, default=Path("./results"))
    parser.add_argument(
        "--inference-free",
        action="store_true",
        dest="inference_free",
        help="Sparse models only: replace the query route with a SparseStaticEmbedding.",
    )
    parser.add_argument(
        "--max-active-dims",
        type=int,
        default=None,
        help="Sparse models only: keep only the k highest-weighted dimensions of "
        "each query/document embedding. Default: unpruned.",
    )
    parser.add_argument(
        "--save-to-cpu",
        action="store_true",
        help="Sparse models only: keep query/document embeddings on the CPU so the "
        "sparse score product runs in system RAM instead of GPU memory (fixes CUDA "
        "OOM in similarity; encoding itself still uses the GPU).",
    )
    return parser.parse_args()


def build_tasks(config_path: Path) -> list:
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    excluded = set(config["excluded_tasks"]) | {"WebFAQRetrieval"}

    tasks = mteb.get_tasks(
        languages=config["languages"],
        modalities=config["modalities"],
        task_types=config["task_types"],
        exclusive_language_filter=True,
        eval_splits=["test"],
    )
    tasks = [t for t in tasks if t.metadata.name not in excluded]

    # Not in MTEB's task registry, so mteb.get_tasks() never returns them, and
    # they are not language-filtered by it either: apply the same filter here,
    # otherwise bilingual tasks (e.g. JaQuAD: it + ja) also run their non-Italian
    # subset and the averaged score no longer matches the Italian-only one.
    tasks += [
        t.filter_languages(
            config["languages"],
            exclusive_language_filter=True,
        )
        for t in [
            CulturaVivaRetrieval(),
            MMarcoITRetrieval(),
            NorQuADRetrieval(),
            JaQuADRetrieval(),
            SberQuADRetrieval(),
            GermanDPRRetrieval(),
        ]
    ]
    return tasks


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be a positive integer.")

    model_name = args.model
    tasks = build_tasks(args.config_path)
    # tasks = [
    #     t
    #     for t in build_tasks(args.config_path)
    #     if t.metadata.name == "MuPLeR-retrieval"
    # ]
    # print(tasks)

    if args.inference_free or Path(model_name).is_dir():
        # MTEB can't resolve these from the Hub: local checkpoints have no Hub
        # metadata (no loader), and the inference-free ablation needs a custom
        # query route. Load the SparseEncoder ourselves and wrap it.
        model = SparseEncoderWrapper(load_sparse_model(model_name, args.inference_free))
        # Keep the original name so results are stored under the right folder.
        model.mteb_model_meta = model.mteb_model_meta.model_copy(
            update={"name": model_name}
        )
    else:
        model = mteb.get_model(model_name)

    encode_kwargs = {"batch_size": args.batch_size}
    if args.max_active_dims is not None:
        encode_kwargs["max_active_dims"] = args.max_active_dims

    if args.save_to_cpu:
        # Forwarded by MTEB to SparseEncoder.encode(save_to_cpu=...).
        encode_kwargs["save_to_cpu"] = True

    safe_model_name = model_name.replace("/", "__")
    prediction_folder = args.results_dir / safe_model_name / "prediction_folder"
    prediction_folder.mkdir(parents=True, exist_ok=True)

    results = mteb.evaluate(
        model=model,
        tasks=tasks,
        cache=mteb.ResultCache("."),
        encode_kwargs=encode_kwargs,
        prediction_folder=str(prediction_folder),
        overwrite_strategy=args.overwrite_strategy,
    )
    print(results)


if __name__ == "__main__":
    main()
