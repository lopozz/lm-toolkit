from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_LANGS = {
    "it": ["ita-Latn"],
    "ja": ["jpn-Jpan"],
}


class JaQuADRetrieval(AbsTaskRetrieval):
    """Custom task: JaQuAD (Japanese) with an added Italian machine translation,
    both hosted under the same repo's language-prefixed subsets."""

    metadata = TaskMetadata(
        name="JaQuADRetrieval",
        description="Japanese question-answering retrieval dataset (JaQuAD), "
        "with an added Italian machine-translated variant.",
        reference="https://huggingface.co/datasets/lopozz/JaQuADRetrieval",
        dataset={
            "path": "lopozz/JaQuADRetrieval",
            "revision": "f69273d7f0d0fe5254e5ae2f41f42a8aa395634d",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
    )
