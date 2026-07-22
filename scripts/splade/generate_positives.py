#!/usr/bin/env python3
"""
Generate CulturaViva-Retrieval-compatible synthetic IR data.

Input:
    corpus.jsonl
    Each row should contain at least:
        {"_id": "...", "text": "..."}
    Optional:
        {"title": "..."}

Output:
    out/
      corpus/train.jsonl
      queries/train.jsonl
      qrels/train.jsonl

Each input document gets exactly ONE positive query.

Example with vLLM:

    python generate_culturaviva_splade_data.py \
        --input corpus.jsonl \
        --output-dir synthetic_culturaviva \
        --backend vllm \
        --base-url http://localhost:8000/v1 \
        --api-key token-abc123 \
        --model Qwen/Qwen2.5-7B-Instruct \
        --case-policy alternate

Example with OpenAI:
    export OPENAI_API_KEY="..."

    python generate_culturaviva_splade_data.py \
        --input corpus.jsonl \
        --output-dir synthetic_culturaviva \
        --backend openai \
        --model gpt-4.1-mini \
        --case-policy alternate
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


ITALIAN_STOPWORDS = {
    "a", "ad", "al", "allo", "ai", "agli", "alla", "alle",
    "con", "col", "coi", "da", "dal", "dallo", "dai", "dagli",
    "dalla", "dalle", "di", "del", "dello", "dei", "degli",
    "della", "delle", "in", "nel", "nello", "nei", "negli",
    "nella", "nelle", "su", "sul", "sullo", "sui", "sugli",
    "sulla", "sulle", "per", "tra", "fra", "il", "lo", "la",
    "i", "gli", "le", "un", "uno", "una", "ma", "ed", "e",
    "o", "che", "chi", "cui", "non", "più", "come", "dove",
    "quando", "quanto", "quale", "quali", "questo", "questa",
    "questi", "queste", "quello", "quella", "quelli", "quelle",
    "sono", "è", "era", "erano", "essere", "ha", "hanno", "avere",
}


SYSTEM_PROMPT = """
Sei un generatore di query sintetiche per addestrare un modello di Information Retrieval in italiano.

Devi generare UNA SOLA query positiva per il documento dato.

Rispondi solo con JSON valido.

Schema obbligatorio:
{
  "query": "...",
}
"""


BM25_EASY_PROMPT = """
Genera UNA query positiva per il seguente documento.

Requisiti:
- La query deve essere facile per BM25.
- Deve condividere diversi termini contenutistici importanti con il documento.
- Deve sembrare una vera query utente, non una lista di keyword.
- La query contiene massimo 15 parole.
- Non inventare informazioni non presenti.

Documento:
\"\"\"
{document}
\"\"\"
"""


BM25_HARD_PROMPT = """
Genera UNA query positiva per il seguente documento.

Requisiti:
- La query deve essere difficile per BM25.
- Usa sinonimi, riformulazioni, iperonimi o termini impliciti.
- Tuttavia deve mantenere 1-2 termini lessicali discriminanti presenti nel documento.
- La query deve essere ancora chiaramente collegata al documento.
- Deve sembrare una vera query utente, non una lista di keyword.
- La query contiene massimo 15 parole.
- Non inventare informazioni non presenti.

Documento:
\"\"\"
{document}
\"\"\"
"""


def lexical_terms(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", text.lower())
    return [
        token
        for token in tokens
        if len(token) > 2 and token not in ITALIAN_STOPWORDS
    ]


def lexical_overlap_ratio(query: str, doc: str) -> float:
    q = set(lexical_terms(query))
    d = set(lexical_terms(doc))
    if not q:
        return 0.0
    return len(q & d) / len(q)


def parse_json_object(raw: str) -> Dict[str, Any]:
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output:\n{raw}")

    return json.loads(raw[start : end + 1])


def get_client(args: argparse.Namespace) -> OpenAI:
    if args.backend == "openai":
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OpenAI API key.")
        return OpenAI(api_key=api_key)

    return OpenAI(
        base_url=args.base_url,
        api_key=args.api_key or "token-abc123",
    )


@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(4))
def call_llm(
    client: OpenAI,
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    strict_json: bool,
) -> Dict[str, Any]:
    kwargs = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }

    # Some vLLM-served models support this, some do not.
    # Disable with --no-strict-json if your server errors.
    if strict_json:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    return parse_json_object(content)


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def choose_case(policy: str, index: int) -> str:
    if policy == "easy":
        return "bm25_easy"
    if policy == "hard":
        return "bm25_hard_expansion"
    if policy == "alternate":
        return "bm25_easy" if index % 2 == 0 else "bm25_hard_expansion"
    if policy == "random":
        return random.choice(["bm25_easy", "bm25_hard_expansion"])
    raise ValueError(f"Unknown case policy: {policy}")


def validate_query(query: str, document: str, case_name: str) -> bool:
    if len(query) < 8:
        return False

    overlap = lexical_overlap_ratio(query, document)

    if case_name == "bm25_easy":
        return overlap >= 0.30

    if case_name == "bm25_hard_expansion":
        # Must have some lexical anchors, but should not look copied.
        return 0.08 <= overlap <= 0.80

    return False


def generate_one_query(
    client: OpenAI,
    args: argparse.Namespace,
    document: str,
    case_name: str,
) -> Optional[Dict[str, Any]]:
    if case_name == "bm25_easy":
        prompt = BM25_EASY_PROMPT.format(document=document)
    else:
        prompt = BM25_HARD_PROMPT.format(document=document)

    data = call_llm(
        client=client,
        model=args.model,
        user_prompt=prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        strict_json=args.strict_json,
    )

    query = data["query"].strip()

    # if not validate_query(query, document, case_name):
    #     return None

    return {
        "query": query,
        # "case": case_name,
        # "lexical_terms_to_preserve": data.get("lexical_terms_to_preserve", []),
        # "useful_expansion_terms": data.get("useful_expansion_terms", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--backend", choices=["openai", "vllm"], default="vllm")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=None)

    parser.add_argument(
        "--case-policy",
        choices=["easy", "hard", "alternate", "random"],
        default="alternate",
        help="How to choose the single query type per document.",
    )

    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-json", action="store_true", default=True)
    parser.add_argument("--no-strict-json", dest="strict_json", action="store_false")

    args = parser.parse_args()
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    corpus_dir = out_dir / "corpus"
    queries_dir = out_dir / "queries"
    qrels_dir = out_dir / "qrels"

    corpus_dir.mkdir(parents=True, exist_ok=True)
    queries_dir.mkdir(parents=True, exist_ok=True)
    qrels_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = corpus_dir / "train.jsonl"
    queries_path = queries_dir / "train.jsonl"
    qrels_path = qrels_dir / "train.jsonl"

    # Start fresh.
    for path in [corpus_path, queries_path, qrels_path]:
        if path.exists():
            path.unlink()

    client = get_client(args)

    docs = list(read_jsonl(args.input))
    if args.max_docs is not None:
        docs = docs[: args.max_docs]

    kept = 0
    skipped = 0

    for i, row in enumerate(tqdm(docs, desc="Generating one query per document")):
        
        doc_id = str(row["_id"])
        title = row["title"]
        text = row["text"]

        case_name = choose_case(args.case_policy, i)

        try:
            generated = generate_one_query(
                client=client,
                args=args,
                document=text,
                case_name=case_name,
            )

            if generated is None:
                skipped += 1
                continue

            query_id = f"q_{kept}"

            append_jsonl(
                corpus_path,
                {
                    "_id": doc_id,
                    "title": title,
                    "text": text,
                },
            )

            append_jsonl(
                queries_path,
                {
                    "_id": query_id,
                    "text": generated["query"],
                },
            )

            append_jsonl(
                qrels_path,
                {
                    "query-id": query_id,
                    "corpus-id": doc_id,
                    "score": 1,
                },
            )

            kept += 1

        except Exception as e:
            skipped += 1
            print(f"[WARN] failed doc_id={doc_id}: {e}")

    print(f"Done. Kept={kept}, skipped={skipped}")
    print("Wrote:")
    print(f"  {corpus_path}")
    print(f"  {queries_path}")
    print(f"  {qrels_path}")


if __name__ == "__main__":
    main()