"""
Dataset-distribution analysis for SPLADE retrieval datasets: token
length, vocabulary size, and truncation rate for queries and documents. Uses
only a tokenizer, never the sparse encoder itself -- meant as a first look
at a dataset's shape before running the (much more expensive) encoder-based
analysis in lm_toolkit/benchmarks/sparse_analysis.py.

Typical usage:

    python3 scripts/splade/analyze_dataset_stats.py \\
      --tokenizer nickprock/splade-bert-base-italian-xxl-uncased-cv \\
      --task-name MuPLeR-retrieval --language it
"""

import argparse

import pandas as pd
from transformers import AutoTokenizer

from lm_toolkit.benchmarks.sparse_analysis import (
    RetrievalTask,
    add_dataset_quantile_buckets,
    load_corpus,
    load_queries,
    p10,
    p90,
    transpose_by_stat,
)

_SENTINEL_MAX_LENGTH = 100_000  # unset tokenizers report an unusably large sentinel


def resolve_max_length(tokenizer: AutoTokenizer) -> int:
    max_length = tokenizer.model_max_length
    if not max_length or max_length > _SENTINEL_MAX_LENGTH:
        max_length = 512
    return max_length


def text_length_metrics(text: str, tokenizer: AutoTokenizer, encoder_max_length: int) -> dict:
    raw_ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    effective_ids = raw_ids[: encoder_max_length - 2]
    return {
        "raw_token_length": len(raw_ids),
        "effective_token_length": len(effective_ids),
        "vocab_size": len(set(effective_ids)),
        "was_truncated": int(len(raw_ids) > len(effective_ids)),
    }


def build_length_frame(
    texts_by_id: dict[str, str], tokenizer: AutoTokenizer, encoder_max_length: int
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"text_id": text_id, **text_length_metrics(text, tokenizer, encoder_max_length)}
            for text_id, text in texts_by_id.items()
        ]
    )


def summarize_overall_lengths(frame: pd.DataFrame) -> pd.DataFrame:
    length_metrics = (
        ("effective_token_length", "effective_length"),
        ("raw_token_length", "raw_length"),
        ("vocab_size", "vocab_size"),
        ("was_truncated", "truncation_rate"),
    )
    records = [
        {
            "metric": "documents",
            "p10": float("nan"),
            "mean": len(frame),
            "median": float("nan"),
            "p90": float("nan"),
        }
    ]
    records.extend(
        {
            "metric": label,
            "p10": p10(frame[column]),
            "mean": frame[column].mean(),
            "median": frame[column].median(),
            "p90": p90(frame[column]),
        }
        for column, label in length_metrics
    )
    return pd.DataFrame(records)


def summarize_bucket_lengths(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby("bucket", observed=True, sort=True)
        .agg(
            documents=("text_id", "count"),
            effective_length_p10=("effective_token_length", p10),
            effective_length_mean=("effective_token_length", "mean"),
            effective_length_median=("effective_token_length", "median"),
            effective_length_p90=("effective_token_length", p90),
            raw_length_p10=("raw_token_length", p10),
            raw_length_mean=("raw_token_length", "mean"),
            raw_length_median=("raw_token_length", "median"),
            raw_length_p90=("raw_token_length", p90),
            vocab_size_p10=("vocab_size", p10),
            vocab_size_mean=("vocab_size", "mean"),
            vocab_size_median=("vocab_size", "median"),
            vocab_size_p90=("vocab_size", p90),
            truncation_rate=("was_truncated", "mean"),
        )
        .reset_index()
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer name or local path. No sparse encoder is loaded.",
    )
    parser.add_argument("--task-name", default="MuPLeR-retrieval")
    parser.add_argument("--language", default="it")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-buckets", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = RetrievalTask(task_name=args.task_name, language=args.language, split=args.split)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    encoder_max_length = resolve_max_length(tokenizer)

    texts_by_label = {
        "document": load_corpus(task),
        "query": load_queries(task),
    }

    for label, texts_by_id in texts_by_label.items():
        frame = build_length_frame(texts_by_id, tokenizer, encoder_max_length)
        frame = add_dataset_quantile_buckets(frame, args.num_buckets)

        print(f"\n{task.task_name} -- {label} overall summary")
        print(summarize_overall_lengths(frame).round(3).to_string(index=False))

        print(f"\n{task.task_name} -- {label} bucketed summary")
        print(transpose_by_stat(summarize_bucket_lengths(frame), "bucket").to_string(index=False))


if __name__ == "__main__":
    main()
