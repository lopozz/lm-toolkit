"""
Shared utilities for inspecting and encoding with SPLADE-style SparseEncoder
models: chunk-and-max-merge document encoding, per-document sparse-activation
metrics, and encoder-aware token-length helpers.

Chunk-and-max-merge (`apply_super_encode_document`)
----------------------------------------------------
BERT-based SPLADE backbones have a fixed context window (learned position
embeddings, typically 512 tokens) and silently truncate longer documents.
SPLADE's own pooling is already a max over token positions within one forward
pass; taking a max again across chunk-level outputs extends that same
operation across chunks instead of improvising a new aggregation scheme.

It rebinds `model.encode_document` on a live SparseEncoder instance. Callers
that resolve `encode_document` via attribute lookup pick up the patched
behavior automatically, with no call-site changes required. Documents that
already fit within the window pass through as a single chunk, so there is no
behavior change for the common case.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import string

import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, Value, concatenate_datasets
from sentence_transformers import SparseEncoder
from sentence_transformers.util import select_max_active_dims
from transformers import AutoTokenizer

from sentence_transformers.sparse_encoder.losses import (
    SparseMultipleNegativesRankingLoss,
    FlopsLoss,
)


_SENTINEL_MAX_LENGTH = 100_000  # unset tokenizers report an unusably large sentinel


def _default_window_tokens(model: SparseEncoder, margin: int = 2) -> int:
    """Leaves room for the [CLS]/[SEP] tokens the real encode call re-adds per chunk."""
    max_length = getattr(model, "max_seq_length", None) or model.tokenizer.model_max_length
    if not max_length or max_length > _SENTINEL_MAX_LENGTH:
        max_length = 512
    return max_length - margin


def _split_into_windows(text: str, tokenizer: Any, window_tokens: int) -> list[str]:
    token_ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    if len(token_ids) <= window_tokens:
        return [text]

    return [
        tokenizer.decode(token_ids[start : start + window_tokens], skip_special_tokens=True)
        for start in range(0, len(token_ids), window_tokens)
    ]


def apply_super_encode_document(
    model: SparseEncoder,
    window_tokens: int | None = None,
) -> SparseEncoder:
    if getattr(model, "_super_encode_document", False):
        return model

    real_encode_document = model.encode_document
    tokenizer = model.tokenizer
    resolved_window_tokens = window_tokens or _default_window_tokens(model)

    def super_encode_document(inputs: list[str] | str, **kwargs: Any) -> torch.Tensor:
        single_input = isinstance(inputs, str)
        texts = [inputs] if single_input else list(inputs)
        batch_size = kwargs.get("batch_size", 32)
        convert_to_sparse_tensor = kwargs.get("convert_to_sparse_tensor", True)
        max_active_dims = kwargs.pop("max_active_dims", None)

        chunk_kwargs = dict(kwargs)
        chunk_kwargs["convert_to_sparse_tensor"] = False
        chunk_kwargs["show_progress_bar"] = False

        merged_vectors: list[torch.Tensor] = [None] * len(texts)  # type: ignore[list-item]

        # Bounds memory to one caller-batch's worth of chunks at a time,
        # regardless of how large the full corpus/inputs list is.
        for start in range(0, len(texts), batch_size):
            sub_texts = texts[start : start + batch_size]

            chunk_texts: list[str] = []
            chunk_owner: list[int] = []
            for owner, text in enumerate(sub_texts):
                windows = _split_into_windows(text, tokenizer, resolved_window_tokens)
                chunk_texts.extend(windows)
                chunk_owner.extend([owner] * len(windows))

            chunk_vectors = real_encode_document(chunk_texts, **chunk_kwargs)

            for owner in range(len(sub_texts)):
                owner_indices = [
                    index for index, chunk_index in enumerate(chunk_owner) if chunk_index == owner
                ]
                merged_vectors[start + owner] = chunk_vectors[owner_indices].max(dim=0).values

        stacked = torch.stack(merged_vectors)

        if max_active_dims is not None:
            stacked = select_max_active_dims(stacked, max_active_dims)
        if convert_to_sparse_tensor:
            stacked = stacked.to_sparse()

        return stacked[0] if single_input else stacked

    model.encode_document = super_encode_document
    model._super_encode_document = True
    return model



ITALIAN_STOPWORDS = {
    "a", "ad", "al", "allo", "ai", "agli", "all", "agl", "alla", "alle",
    "con", "col", "coi",
    "da", "dal", "dallo", "dai", "dagli", "dall", "dagl", "dalla", "dalle",
    "di", "del", "dello", "dei", "degli", "dell", "degl", "della", "delle",
    "in", "nel", "nello", "nei", "negli", "nell", "negl", "nella", "nelle",
    "su", "sul", "sullo", "sui", "sugli", "sull", "sugl", "sulla", "sulle",
    "per", "tra", "fra",
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
    "e", "ed", "o", "od", "ma", "se", "che", "chi", "cui",
    "questo", "questa", "questi", "queste", "quello", "quella", "quelli", "quelle",
    "sono", "sei", "è", "era", "erano", "essere", "stato", "stata", "stati", "state",
    "ho", "hai", "ha", "abbiamo", "avete", "hanno", "avere",
    "mi", "ti", "si", "ci", "vi", "me", "te", "lui", "lei", "noi", "voi", "loro",
    "non", "più", "anche", "come", "dove", "quando", "perché",
}


class LexicalAwareSpladeLoss(nn.Module):
    """
    SPLADE ranking + FLOPS sparsity + lexical retention + stopword/punctuation suppression.

    Assumes sentence_features[0] is the query route and sentence_features[1:] are document routes,
    matching your router_mapping:
        "query"  -> "query"
        "answer" -> "document"
    """

    def __init__(
        self,
        model,
        tokenizer,
        main_loss=None,
        query_regularizer_weight=3e-5,
        document_regularizer_weight=1e-5,
        query_retention_weight=0.03,
        document_retention_weight=0.10,
        blocked_token_weight=0.05,
        lexical_margin=0.05,
        stopwords=None,
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer

        self.main_loss = main_loss or SparseMultipleNegativesRankingLoss(model=model)

        self.query_regularizer = FlopsLoss(model)
        self.document_regularizer = FlopsLoss(model)
        self.query_regularizer_weight = query_regularizer_weight
        self.document_regularizer_weight = document_regularizer_weight

        self.query_retention_weight = query_retention_weight
        self.document_retention_weight = document_retention_weight
        self.blocked_token_weight = blocked_token_weight
        self.lexical_margin = lexical_margin

        blocked_ids = self._build_blocked_token_ids(stopwords or ITALIAN_STOPWORDS)
        self.register_buffer(
            "blocked_token_ids",
            torch.tensor(sorted(blocked_ids), dtype=torch.long),
            persistent=False,
        )

    def _build_blocked_token_ids(self, stopwords):
        blocked = set()

        # Special tokens
        for tok_id in self.tokenizer.all_special_ids:
            if tok_id is not None:
                blocked.add(int(tok_id))

        # Italian stopwords
        for word in stopwords:
            ids = self.tokenizer.encode(word, add_special_tokens=False)
            blocked.update(int(i) for i in ids)

        # Punctuation-like vocab entries
        vocab = self.tokenizer.get_vocab()
        punct_chars = set(string.punctuation) | {
            "«", "»", "“", "”", "‘", "’", "…", "–", "—", "·", "•"
        }

        for token, tok_id in vocab.items():
            clean = (
                token
                .replace("##", "")
                .replace("▁", "")
                .replace("Ġ", "")
                .strip()
            )

            if clean and all(ch in punct_chars for ch in clean):
                blocked.add(int(tok_id))

        return blocked

    def _is_blocked(self, input_ids):
        if self.blocked_token_ids.numel() == 0:
            return torch.zeros_like(input_ids, dtype=torch.bool)

        return torch.isin(input_ids, self.blocked_token_ids.to(input_ids.device))

    def _lexical_retention_loss(self, embedding, sentence_feature):
        """
        Penalize the model when original content-token ids have weight < lexical_margin.

        embedding: [batch, vocab]
        input_ids: [batch, seq_len]
        """
        input_ids = sentence_feature["input_ids"]
        attention_mask = sentence_feature.get("attention_mask")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        blocked = self._is_blocked(input_ids)
        content_mask = attention_mask & ~blocked

        if content_mask.sum() == 0:
            return embedding.sum() * 0.0

        token_weights = embedding.gather(dim=1, index=input_ids)
        loss = F.relu(self.lexical_margin - token_weights)

        return loss[content_mask].mean()

    def _blocked_token_loss(self, embedding):
        """
        Penalize assigning mass to stopwords, punctuation, and special tokens.
        """
        if self.blocked_token_ids.numel() == 0:
            return embedding.sum() * 0.0

        blocked_ids = self.blocked_token_ids.to(embedding.device)
        blocked_weights = embedding[:, blocked_ids]

        return blocked_weights.pow(2).mean()

    def forward(self, sentence_features, labels=None):
        embeddings = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]

        losses = {}

        base_loss = self.main_loss.compute_loss_from_embeddings(embeddings, labels)
        if isinstance(base_loss, dict):
            losses.update(base_loss)
        else:
            losses["base_loss"] = base_loss

        query_embedding = embeddings[0]
        document_embeddings = embeddings[1:]

        losses["query_regularizer_loss"] = (
            self.query_regularizer.compute_loss_from_embeddings(query_embedding)
            * self.query_regularizer_weight
        )

        losses["document_regularizer_loss"] = (
            self.document_regularizer.compute_loss_from_embeddings(
                torch.cat(document_embeddings, dim=0)
            )
            * self.document_regularizer_weight
        )

        losses["query_retention_loss"] = (
            self._lexical_retention_loss(query_embedding, sentence_features[0])
            * self.query_retention_weight
        )

        doc_retention = 0.0
        for doc_embedding, doc_features in zip(document_embeddings, sentence_features[1:]):
            doc_retention = doc_retention + self._lexical_retention_loss(
                doc_embedding, doc_features
            )

        doc_retention = doc_retention / max(len(document_embeddings), 1)
        losses["document_retention_loss"] = doc_retention * self.document_retention_weight

        blocked_loss = self._blocked_token_loss(query_embedding)
        for doc_embedding in document_embeddings:
            blocked_loss = blocked_loss + self._blocked_token_loss(doc_embedding)

        blocked_loss = blocked_loss / len(embeddings)
        losses["blocked_token_loss"] = blocked_loss * self.blocked_token_weight

        return losses
    

# loss = LexicalAwareSpladeLoss(
#     model=model,
#     tokenizer=mlm_transformer.tokenizer,
#     main_loss=SparseMultipleNegativesRankingLoss(
#         model=model,
#         # Optional but often helpful once batch_size is > 1:
#         directions=("query_to_doc", "doc_to_query"),
#         partition_mode="per_direction",
#     ),

#     # Much lower than your current 3e-3.
#     # 3e-3 is very aggressive and can easily teach the document encoder
#     # to delete many useful original terms.
#     document_regularizer_weight=1e-5,

#     # You use inference-free static query embeddings, so this can stay small.
#     # Do not set it high unless queries become too verbose.
#     query_regularizer_weight=3e-5,

#     # Stronger on documents: documents should preserve lexical evidence.
#     document_retention_weight=0.10,

#     # Weaker on queries: queries should preserve key terms but may expand more.
#     query_retention_weight=0.03,

#     # Suppress stopwords/punctuation.
#     blocked_token_weight=0.05,

#     # Minimum desired SPLADE weight for original content tokens.
#     lexical_margin=0.05,
# )


# Dataset combination
# --------------------
# Combine one or more local MTEB-style retrieval datasets (corpus/queries/qrels
# parquet triples under data/<name>/, e.g. produced by translator.py) into a
# single split, with an optional per-dataset document token-length filter.
#
# _id values are namespaced with the dataset name before concatenation, since
# different datasets commonly reuse generic ids (e.g. "c0", "q0").
#
# Typical usage from a notebook:
#
#     from lm_toolkit.splade import DatasetSpec, combine_datasets
#
#     combined = combine_datasets(
#         [
#             DatasetSpec("mmarco_it_75k", max_tokens=400),
#             DatasetSpec("sberquad_mteb_itquad"),
#         ],
#         split="train",
#     )
#     combined["corpus"], combined["queries"], combined["qrels"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_TOKENIZER = "nickprock/splade-bert-base-italian-xxl-uncased-cv"


@dataclass
class DatasetSpec:
    name: str
    min_tokens: int | None = None
    max_tokens: int | None = None


def _load_split(data_dir: Path, dataset_name: str, split: str) -> tuple[Dataset, Dataset, Dataset]:
    dataset_dir = data_dir / dataset_name
    corpus = Dataset.from_parquet(str(dataset_dir / f"{split}_corpus.parquet"))
    queries = Dataset.from_parquet(str(dataset_dir / f"{split}_queries.parquet"))

    # Different source datasets can emit "score" as int32 vs int64, which
    # blocks concatenate_datasets(); normalize to a single type here.
    qrels = Dataset.from_parquet(str(dataset_dir / f"{split}_qrels.parquet"))
    qrels = qrels.cast_column("score", Value("int64"))

    return corpus, queries, qrels


def _prefix_ids(
    corpus: Dataset, queries: Dataset, qrels: Dataset, dataset_name: str
) -> tuple[Dataset, Dataset, Dataset]:
    prefix = f"{dataset_name}::"

    corpus = corpus.map(
        lambda batch: {"_id": [prefix + id_ for id_ in batch["_id"]]},
        batched=True,
    )
    queries = queries.map(
        lambda batch: {"_id": [prefix + id_ for id_ in batch["_id"]]},
        batched=True,
    )
    qrels = qrels.map(
        lambda batch: {
            "query-id": [prefix + id_ for id_ in batch["query-id"]],
            "corpus-id": [prefix + id_ for id_ in batch["corpus-id"]],
        },
        batched=True,
    )

    return corpus, queries, qrels


def _filter_by_token_length(
    corpus: Dataset,
    queries: Dataset,
    qrels: Dataset,
    tokenizer: AutoTokenizer,
    min_tokens: int | None,
    max_tokens: int | None,
) -> tuple[Dataset, Dataset, Dataset]:
    doc_token_lengths = [
        len(ids) for ids in tokenizer(list(corpus["text"]), add_special_tokens=True)["input_ids"]
    ]

    def out_of_range(length: int) -> bool:
        if min_tokens is not None and length < min_tokens:
            return True
        if max_tokens is not None and length > max_tokens:
            return True
        return False

    dropped_doc_ids = {
        doc_id
        for doc_id, length in zip(corpus["_id"], doc_token_lengths)
        if out_of_range(length)
    }

    if not dropped_doc_ids:
        return corpus, queries, qrels

    corpus = corpus.filter(lambda row: row["_id"] not in dropped_doc_ids)
    qrels = qrels.filter(lambda row: row["corpus-id"] not in dropped_doc_ids)

    remaining_query_ids = set(qrels["query-id"])
    queries = queries.filter(lambda row: row["_id"] in remaining_query_ids)

    return corpus, queries, qrels


def combine_datasets(
    specs: list[DatasetSpec],
    split: str,
    data_dir: Path = DATA_DIR,
    tokenizer_name: str = DEFAULT_TOKENIZER,
) -> dict[str, Dataset]:
    tokenizer = None
    if any(spec.min_tokens is not None or spec.max_tokens is not None for spec in specs):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    corpora: list[Dataset] = []
    all_queries: list[Dataset] = []
    all_qrels: list[Dataset] = []

    for spec in specs:
        corpus, queries, qrels = _load_split(data_dir, spec.name, split)
        corpus, queries, qrels = _prefix_ids(corpus, queries, qrels, spec.name)

        if spec.min_tokens is not None or spec.max_tokens is not None:
            corpus, queries, qrels = _filter_by_token_length(
                corpus, queries, qrels, tokenizer, spec.min_tokens, spec.max_tokens
            )

        print(
            f"[{spec.name}] {split}: {len(corpus)} documents, "
            f"{len(queries)} queries, {len(qrels)} qrels"
        )

        corpora.append(corpus)
        all_queries.append(queries)
        all_qrels.append(qrels)

    return {
        "corpus": concatenate_datasets(corpora),
        "queries": concatenate_datasets(all_queries),
        "qrels": concatenate_datasets(all_qrels),
    }