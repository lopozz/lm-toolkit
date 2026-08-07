"""Single source of truth for how to load each custom/local retrieval task's
corpus, queries, and qrels -- shared between scripts/splade/evaluate_mteb.py
(ranking quality) and lm_toolkit/benchmarks/sparse_analysis.py (activation
behavior), so the two don't maintain separate, drifting copies of the same
per-task-name special-casing.

Two kinds of task:

- Hosted (HOSTED_TASK_CONFIGS): read from the Hub via `datasets.load_dataset`.
  Most use language-prefixed configs (it-corpus/it-queries/it-qrels, or
  no-corpus/... for the original-language subset); a task's "hf_subset" entry
  is either a fixed override (CulturaViva-Retrieval's unprefixed "default"
  config; WebFAQRetrieval's ISO 639-3 "ita" configs) or None, meaning the
  caller's requested language is used as-is (the normal case).
- Local (LOCAL_TASK_DIRS): parquet-only, never pushed to the Hub, read
  directly from a local directory.
"""

from pathlib import Path

from datasets import Dataset, load_dataset

HOSTED_TASK_CONFIGS = {
    "CulturaViva-Retrieval": {
        "dataset_path": "lopozz/CulturaViva-Retrieval",
        "hf_subset": "default",
        "id_column": "_id",
    },
    "WebFAQRetrieval": {
        "dataset_path": "mteb/WebFAQRetrieval",
        "hf_subset": "ita",
        "id_column": "id",
    },
    "WikipediaRetrievalMultilingual": {
        "dataset_path": "mteb/WikipediaRetrievalMultilingual",
        "hf_subset": None,
        "id_column": "_id",
    },
    "MuPLeR-retrieval": {
        "dataset_path": "mteb/MuPLeR-retrieval",
        "hf_subset": None,
        "id_column": "id",
    },
    "NorQuADRetrieval": {
        "dataset_path": "lopozz/NorQuADRetrieval",
        "hf_subset": None,
        "id_column": "_id",
    },
    "JaQuADRetrieval": {
        "dataset_path": "lopozz/JaQuADRetrieval",
        "hf_subset": None,
        "id_column": "_id",
    },
    "SberQuADRetrieval": {
        "dataset_path": "lopozz/SberQuADRetrieval",
        "hf_subset": None,
        "id_column": "_id",
    },
    "GermanDPRRetrieval": {
        "dataset_path": "lopozz/GermanDPRRetrieval",
        "hf_subset": None,
        "id_column": "_id",
    },
}

LOCAL_TASK_DIRS = {
    "MMarco-IT-Retrieval": Path("data/mmarco_it_dev_small_50k_len50_400"),
    "MMarco-IT-2-Retrieval": Path("data/mmarco_it_dev_small_50k_len1_500"),
}

ALL_TASK_NAMES = frozenset(HOSTED_TASK_CONFIGS) | frozenset(LOCAL_TASK_DIRS)


def resolve_hf_subset(task_name: str, language: str = "it") -> str:
    if task_name in LOCAL_TASK_DIRS:
        return "default"

    fixed_subset = HOSTED_TASK_CONFIGS[task_name]["hf_subset"]
    return fixed_subset if fixed_subset is not None else language


def resolve_id_column(task_name: str) -> str:
    if task_name in LOCAL_TASK_DIRS:
        return "_id"

    return HOSTED_TASK_CONFIGS[task_name]["id_column"]


def load_retrieval_dataset(
    task_name: str,
    subset: str,
    split: str,
    language: str = "it",
) -> Dataset:
    """Loads one of a task's "corpus", "queries", or "qrels" datasets."""
    if task_name in LOCAL_TASK_DIRS:
        local_dir = LOCAL_TASK_DIRS[task_name]
        return Dataset.from_parquet(str(local_dir / f"{split}_{subset}.parquet"))

    dataset_path = HOSTED_TASK_CONFIGS[task_name]["dataset_path"]
    hf_subset = resolve_hf_subset(task_name, language)
    prefix = "" if hf_subset == "default" else f"{hf_subset}-"

    return load_dataset(path=dataset_path, name=f"{prefix}{subset}", split=split)
