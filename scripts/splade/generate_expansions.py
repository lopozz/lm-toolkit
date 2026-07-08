import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import bm25s
from rich.progress import track

from sentence_transformers import SparseEncoder


# =============================================================================
# Configuration
# =============================================================================

TEXTS = ['Chi mantiene il potere di firma congiunta sul conto fruttifero a doppia firma per assicurare la corretta erogazione?',
 'Quale studio accademico commissionato dal Forum economico mondiale riportò maggiori perdite di occupazione per le acquisizioni rispetto a imprese confrontate?',
 'Perché i ministri hanno sollecitato riallocare il sostegno pubblico dai salvataggi settoriali a programmi ampi per riequilibrare il mercato?',
 "Entro quanto tempo le parti interessate devono manifestarsi dopo la pubblicazione dell'avviso per preservare diritti procedurali e presentare richieste individuali?",
 "Quale bando richiede l'istituzione di un elenco europeo di indirizzi URL contenenti immagini pedopornografiche accessibile alle autorità di polizia nazionali?",
 'Quale valutazione collega classificazioni di entità a fonti di reddito e posizione finanziaria, enfatizzando indebitamento e soglia rischio basso/alto?',
 "Come i pagamenti anticipati per l'accesso riducono inefficienze allocazione scorte permettendo ai produttori di competere e segnalare fiducia nel lancio?",
 'Come calcolare la compensazione per operatore non selezionato via gara, paragonato a impresa ipotetica efficiente dotata di mezzi di trasporto?',
 'Come può una riduzione tariffaria imposta dal governo, finanziata da prelievo parafiscale a fondo statale, avvantaggiare ingiustamente solo tre società?',
 "Perché il Tribunale ha evitato di scegliere tra due opzioni di calcolo e ha ignorato l'anzianità concentrandosi sulla retribuzione base?"]


SPLADE_MODELS = [
    "nickprock/splade-bert-base-italian-xxl-uncased-cv",
    # "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
    "models/splade-bert-base-italian-xxl-uncased-cv",
]


RUN_BM25 = True

BM25_MODEL_NAME = "mteb/baseline-bm25s"

OUTPUT_PATH = Path("expansions.json")

# Use "doc" for document expansion.
# Use "query" if you want to inspect query-side expansion for SPLADE.
SPLADE_MODE = "doc"

TOP_N = -1

ENSURE_ASCII = False


# =============================================================================
# Shared helpers
# =============================================================================

Expansion = List[Tuple[str, float]]


def write_json(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=ENSURE_ASCII,
            indent=2,
        )


def rows_from_expansions(
    texts: List[str],
    mode: str,
    expansions: List[Expansion],
) -> List[Dict[str, Any]]:
    rows = []

    for idx, text in enumerate(texts):
        rows.append(
            {
                "id": idx,
                "text": text,
                "mode": mode,
                "expansion": [
                    {
                        "token": token,
                        "weight": float(weight),
                    }
                    for token, weight in expansions[idx]
                ],
            }
        )

    return rows


# =============================================================================
# SPLADE expansion
# =============================================================================

def load_sparse_encoder(model_name: str) -> SparseEncoder:
    """
    Load SparseEncoder on CPU.

    This keeps inference CPU-only even if CUDA is available.
    """
    model = SparseEncoder(model_name, device="cpu")
    model.eval()
    return model


def build_vocab(model: SparseEncoder) -> Dict[int, str]:
    """
    Convert tokenizer vocabulary from token -> id to id -> token.
    """
    return {idx: token for token, idx in model.tokenizer.get_vocab().items()}


def splade_expand_text(
    text: str,
    model: SparseEncoder,
    vocab: Dict[int, str],
    mode: str = "doc",
    top_n: int = -1,
) -> Expansion:
    """
    Expand a single text using a SparseEncoder model.

    mode="query" uses encode_query.
    mode="doc" uses encode_document.
    """
    with torch.no_grad():
        if mode == "query":
            vector = model.encode_query(
                [text],
                convert_to_sparse_tensor=False,
            )[0]
        else:
            vector = model.encode_document(
                [text],
                convert_to_sparse_tensor=False,
            )[0]

    if not torch.is_tensor(vector):
        vector = torch.tensor(vector)

    weights = vector.cpu().tolist()

    expansion = []

    for idx, weight in enumerate(weights):
        if weight > 0:
            token = vocab.get(idx, f"[UNK_{idx}]")
            expansion.append((token, float(weight)))

    expansion = sorted(
        expansion,
        key=lambda item: item[1],
        reverse=True,
    )

    if top_n != -1:
        expansion = expansion[:top_n]

    return expansion


def run_splade_model(
    texts: List[str],
    model_name: str,
    mode: str = "doc",
    top_n: int = -1,
) -> List[Dict[str, Any]]:
    model = load_sparse_encoder(model_name)
    vocab = build_vocab(model)

    expansions = []

    for text in track(
        texts,
        description=f"[SPLADE] {model_name}",
    ):
        expansion = splade_expand_text(
            text=text,
            model=model,
            vocab=vocab,
            mode=mode,
            top_n=top_n,
        )
        expansions.append(expansion)

    return rows_from_expansions(
        texts=texts,
        mode=mode,
        expansions=expansions,
    )


# =============================================================================
# BM25 expansion
# =============================================================================

def build_bm25_retriever(texts: List[str]) -> bm25s.BM25:
    corpus_tokens = bm25s.tokenize(texts)

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    return retriever


def bm25_expand_document(
    text: str,
    doc_id: int,
    retriever: bm25s.BM25,
    n_docs: int,
    top_n: int = -1,
) -> Expansion:
    """
    BM25 document expansion.

    For each unique token in the document, score that token as a query
    against the whole corpus, then keep this document's BM25 score.
    """
    tokens = bm25s.tokenize(
        [text],
        return_ids=False,
    )[0]

    expansion = []

    for token in sorted(set(tokens)):
        query_tokens = bm25s.tokenize([token])

        doc_ids, scores = retriever.retrieve(
            query_tokens,
            k=n_docs,
        )

        score_by_doc = dict(
            zip(
                doc_ids[0].tolist(),
                scores[0].tolist(),
            )
        )

        weight = float(score_by_doc.get(doc_id, 0.0))

        if weight > 0:
            expansion.append((token, weight))

    expansion = sorted(
        expansion,
        key=lambda item: item[1],
        reverse=True,
    )

    if top_n != -1:
        expansion = expansion[:top_n]

    return expansion


def run_bm25_model(
    texts: List[str],
    model_name: str = BM25_MODEL_NAME,
    top_n: int = -1,
) -> List[Dict[str, Any]]:
    retriever = build_bm25_retriever(texts)
    n_docs = len(texts)

    expansions = []

    for doc_id, text in track(
        list(enumerate(texts)),
        description=f"[BM25] {model_name}",
    ):
        expansion = bm25_expand_document(
            text=text,
            doc_id=doc_id,
            retriever=retriever,
            n_docs=n_docs,
            top_n=top_n,
        )
        expansions.append(expansion)

    return rows_from_expansions(
        texts=texts,
        mode="doc",
        expansions=expansions,
    )


# =============================================================================
# Main
# =============================================================================

def generate_all_expansions() -> Dict[str, Any]:
    output = {}

    for model_name in SPLADE_MODELS:
        rows = run_splade_model(
            texts=TEXTS,
            model_name=model_name,
            mode=SPLADE_MODE,
            top_n=TOP_N,
        )

        output[model_name] = rows

    if RUN_BM25:
        rows = run_bm25_model(
            texts=TEXTS,
            model_name=BM25_MODEL_NAME,
            top_n=TOP_N,
        )

        output[BM25_MODEL_NAME] = rows

    return output


def main() -> None:
    data = generate_all_expansions()
    write_json(data, OUTPUT_PATH)

    print(f"\nSaved expansions to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()