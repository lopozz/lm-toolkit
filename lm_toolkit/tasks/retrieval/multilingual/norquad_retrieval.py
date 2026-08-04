from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_LANGS = {
    "it": ["ita-Latn"],
    "no": ["nor-Latn"],
}


class NorQuADRetrieval(AbsTaskRetrieval):
    """Custom task: NorQuAD (Norwegian) with an added Italian machine translation,
    both hosted under the same repo's language-prefixed subsets."""

    metadata = TaskMetadata(
        name="NorQuADRetrieval",
        description="Norwegian question-answering retrieval dataset (NorQuAD), "
        "with an added Italian machine-translated variant.",
        reference="https://huggingface.co/datasets/lopozz/NorQuADRetrieval",
        dataset={
            "path": "lopozz/NorQuADRetrieval",
            "revision": "d079f3d7c70a62da6275e37cda8fdb309eebb1fd",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
    )
