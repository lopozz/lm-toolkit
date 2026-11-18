from lm_toolkit.tasks.retrieval.ita import CulturaVivaRetrieval, MMarcoITRetrieval
from lm_toolkit.tasks.retrieval.multilingual import (
    GermanDPRRetrieval,
    JaQuADRetrieval,
    NorQuADRetrieval,
    SberQuADRetrieval,
)

# Custom tasks that are not in MTEB's registry (so `mteb.get_tasks()` never
# returns them), keyed by MTEB task name so configs can select them by name
# through `custom_tasks`.
CUSTOM_TASKS = {
    task.metadata.name: task
    for task in (
        CulturaVivaRetrieval,
        MMarcoITRetrieval,
        NorQuADRetrieval,
        JaQuADRetrieval,
        SberQuADRetrieval,
        GermanDPRRetrieval,
    )
}

__all__ = ["CUSTOM_TASKS"]
