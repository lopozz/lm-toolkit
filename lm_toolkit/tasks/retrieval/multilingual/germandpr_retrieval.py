from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_LANGS = {
    "it": ["ita-Latn"],
    "de": ["deu-Latn"],
}


class GermanDPRRetrieval(AbsTaskRetrieval):
    """Custom task: GermanDPR (German) with an added Italian machine translation,
    both hosted under the same repo's language-prefixed subsets."""

    metadata = TaskMetadata(
        name="GermanDPRRetrieval",
        description="German question-answering retrieval dataset (GermanDPR), "
        "with an added Italian machine-translated variant.",
        reference="https://huggingface.co/datasets/lopozz/GermanDPRRetrieval",
        dataset={
            "path": "lopozz/GermanDPRRetrieval",
            "revision": "49ddef97ff854677d3565ea1d7687550a5b32d3b",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
    )
