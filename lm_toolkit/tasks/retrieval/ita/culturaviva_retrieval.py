from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class CulturaVivaRetrieval(AbsTaskRetrieval):
    """Custom task: not registered in MTEB, hosted outside the mteb/ org."""

    metadata = TaskMetadata(
        name="CulturaViva-Retrieval",
        description="Italian retrieval dataset covering culturally-grounded, "
        "long-form generated content.",
        reference="https://huggingface.co/datasets/lopozz/CulturaViva-Retrieval",
        dataset={
            "path": "lopozz/CulturaViva-Retrieval",
            "revision": "2347b286719a879cc5129cea5c4c5d3fa813dd1b",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ita-Latn"],
        main_score="ndcg_at_10",
    )
