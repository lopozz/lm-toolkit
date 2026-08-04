from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_LANGS = {
    "it": ["ita-Latn"],
    "ru": ["rus-Cyrl"],
}


class SberQuADRetrieval(AbsTaskRetrieval):
    """Custom task: SberQuAD (Russian) with an added Italian machine translation,
    both hosted under the same repo's language-prefixed subsets."""

    metadata = TaskMetadata(
        name="SberQuADRetrieval",
        description="Russian question-answering retrieval dataset (SberQuAD), "
        "with an added Italian machine-translated variant.",
        reference="https://huggingface.co/datasets/lopozz/SberQuADRetrieval",
        dataset={
            "path": "lopozz/SberQuADRetrieval",
            "revision": "4d205f3e335b2adab50dc2ee1031af6588436960",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
    )
