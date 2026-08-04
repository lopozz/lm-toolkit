from pathlib import Path

from datasets import Dataset, Features, Value
from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MMarcoITRetrieval(AbsTaskRetrieval):
    """Custom task: local parquet-only dataset, not hosted on the Hub.
    Overrides load_data() to read data/mmarco_it_dev_small_50k_len50_400
    directly instead of fetching self.metadata.dataset["path"] from the Hub,
    so this works through both the sparse branch and mteb.evaluate()."""

    metadata = TaskMetadata(
        name="MMarco-IT-Retrieval",
        description="mmarco Italian retrieval dev sample (50k docs, length 50-400).",
        reference="https://huggingface.co/datasets/unicamp-dl/mmarco",
        dataset={
            "path": "local/mmarco_it_dev_small_50k_len50_400",
            "revision": "local",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ita-Latn"],
        main_score="ndcg_at_10",
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        local_dir = Path("data/mmarco_it_dev_small_50k_len50_400")
        split = self.eval_splits[0]

        corpus_ds = Dataset.from_parquet(str(local_dir / f"{split}_corpus.parquet"))
        if "_id" in corpus_ds.column_names:
            corpus_ds = corpus_ds.cast_column("_id", Value("string")).rename_column("_id", "id")

        queries_ds = Dataset.from_parquet(str(local_dir / f"{split}_queries.parquet"))
        if "_id" in queries_ds.column_names:
            queries_ds = queries_ds.cast_column("_id", Value("string")).rename_column("_id", "id")

        qrels_ds = Dataset.from_parquet(str(local_dir / f"{split}_qrels.parquet"))
        qrels_ds = qrels_ds.select_columns(["query-id", "corpus-id", "score"])
        qrels_ds = qrels_ds.cast(
            Features(
                {
                    "query-id": Value("string"),
                    "corpus-id": Value("string"),
                    "score": Value("int32"),
                }
            )
        )
        qrels_ds = qrels_ds.to_polars()
        qrels_dict = {
            query_id[0]: dict(zip(group["corpus-id"], group["score"]))
            for query_id, group in qrels_ds.group_by("query-id", maintain_order=False)
        }

        # Matches RetrievalDatasetLoader.load(): only keep queries that have qrels.
        ids_to_keep = set(qrels_dict.keys())
        indices = [i for i, id_ in enumerate(queries_ds["id"]) if id_ in ids_to_keep]
        queries_ds = queries_ds.select(indices)

        self.dataset = {
            "default": {
                split: {
                    "corpus": corpus_ds,
                    "queries": queries_ds,
                    "relevant_docs": qrels_dict,
                    "top_ranked": None,
                }
            }
        }
        self.dataset_transform(num_proc=num_proc)
        self.data_loaded = True


class MMarcoITRetrieval2(AbsTaskRetrieval):
    """Custom task: local parquet-only dataset, not hosted on the Hub.
    Overrides load_data() to read data/mmarco_it_dev_small_10k_len1_50
    directly instead of fetching self.metadata.dataset["path"] from the Hub,
    so this works through both the sparse branch and mteb.evaluate()."""

    metadata = TaskMetadata(
        name="MMarco-IT-2-Retrieval",
        description="mmarco Italian retrieval dev sample (10k docs, length 50-400).",
        reference="https://huggingface.co/datasets/unicamp-dl/mmarco",
        dataset={
            "path": "local/mmarco_it_dev_small_50k_len1_500",
            "revision": "local",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ita-Latn"],
        main_score="ndcg_at_10",
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        local_dir = Path("data/mmarco_it_dev_small_10k_len1_50")
        split = self.eval_splits[0]

        corpus_ds = Dataset.from_parquet(str(local_dir / f"{split}_corpus.parquet"))
        if "_id" in corpus_ds.column_names:
            corpus_ds = corpus_ds.cast_column("_id", Value("string")).rename_column("_id", "id")

        queries_ds = Dataset.from_parquet(str(local_dir / f"{split}_queries.parquet"))
        if "_id" in queries_ds.column_names:
            queries_ds = queries_ds.cast_column("_id", Value("string")).rename_column("_id", "id")

        qrels_ds = Dataset.from_parquet(str(local_dir / f"{split}_qrels.parquet"))
        qrels_ds = qrels_ds.select_columns(["query-id", "corpus-id", "score"])
        qrels_ds = qrels_ds.cast(
            Features(
                {
                    "query-id": Value("string"),
                    "corpus-id": Value("string"),
                    "score": Value("int32"),
                }
            )
        )
        qrels_ds = qrels_ds.to_polars()
        qrels_dict = {
            query_id[0]: dict(zip(group["corpus-id"], group["score"]))
            for query_id, group in qrels_ds.group_by("query-id", maintain_order=False)
        }

        # Matches RetrievalDatasetLoader.load(): only keep queries that have qrels.
        ids_to_keep = set(qrels_dict.keys())
        indices = [i for i, id_ in enumerate(queries_ds["id"]) if id_ in ids_to_keep]
        queries_ds = queries_ds.select(indices)

        self.dataset = {
            "default": {
                split: {
                    "corpus": corpus_ds,
                    "queries": queries_ds,
                    "relevant_docs": qrels_dict,
                    "top_ranked": None,
                }
            }
        }
        self.dataset_transform(num_proc=num_proc)
        self.data_loaded = True
