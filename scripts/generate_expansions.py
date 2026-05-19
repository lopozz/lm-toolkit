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

TEXTS = ["Per evitare ulteriori danneggiamenti che comporterebbero un calo di quantità di prodotto vendibile e soprattutto un calo di qualità dell'intera partita è determinante eseguire tale lavorazione in tempi brevi, quando il bulbo presenta ancora tutto il suo turgore cellulare: la pratica della pelatura solo così può essere effettuata nelle condizioni fisiologiche del bulbo più idonee e si potranno ottenere i migliori risultati possibili. Per ben evidenziare l'importanza di eseguire sui luoghi di produzione in tempi brevi tutte le fasi di lavorazione e confezionamento, va sottolineato che il bulbo del Cipollotto Nocerino è consumato crudo allo stato fresco e le sue principali caratteristiche (fragranza, brillantezza, delicatezza, sapidità, croccantezza, turgidità) che ne hanno fatto un prodotto unico e di pregio verrebbero irrimediabilmente compromesse con eventuali ulteriori manipolazioni e/o trasferimenti in altri luoghi.",
 "Il CESE condivide l'opportunità di esaminare attentamente la legittimità delle barriere alla mobilità dei consumatori, quali elevate commissioni di chiusura, scarsa trasparenza informativa, strutture contrattuali dei servizi finanziari eccessivamente proiettate sulla creazione di vincoli al cambio di prodotto o fornitore, come avviene in alcuni paesi. Il CESE, tuttavia, sottolinea anche che vi sono vincoli tecnologici e normativi, fiscali e legali, che è spesso difficile superare per attuare le condizioni che permettano la mobilità dei consumatori ai livelli indicati dalla Commissione. Inoltre sussiste il pericolo che la semplificazione delle disposizioni sui servizi finanziari conduca a un abbassamento del livello di protezione dei consumatori. L'abbattimento delle barriere non deve portare a un rincaro dei prodotti né a un peggioramento delle norme di tutela esistenti.",
 "Il CESE si compiace del fatto che il Libro bianco sottolinei la necessità di un coordinamento delle misure di lotta al doping e pone l'accento sull'esigenza che gli Stati membri coordinino le loro iniziative in questo campo con quelle delle organizzazioni internazionali esistenti, soprattutto per evitare doppioni e garantire perciò un impiego più efficiente delle risorse. Attualmente la lotta contro il doping è insufficiente e non riesce a scoraggiare i giovani dal fare uso di sostanze dopanti. A questo scopo il CESE raccomanda l'elaborazione di uno studio sullo stato delle legislazioni nazionali in materia, nonché un'analisi giuridica comparata delle carenze e delle lacune riscontrate.",
 "Il CESE, come già proposto nel parere sul tema dell'efficienza energetica, ritiene che sarebbe utilissimo avere a disposizione un portale web ove le ricerche svolte nell'ambito accademico e le sperimentazioni che vengono realizzate a livello nazionale, nelle regioni e nelle città, possano essere fatte conoscere ad un pubblico più vasto ed in particolare agli amministratori locali. Il CESE ritiene che per ottenere un mix energetico ottimale sia necessario un adeguato mix nel trasporto, incrementando l'efficienza degli idrocarburi e la definizione di priorità nel trasporto. In attesa di poter contare su una produzione efficiente di idrogeno, è indifferibile l'utilizzo dell'elettricità, prodotta da energie rinnovabili. La sfida nel trasporto, in tempi brevi e ove possibile, è di utilizzare sempre più elettricità, prodotta da energie pulite e rinnovabili.",
 "Le varietà Makói vöröshagyma o Makói hagyma sono classificate, imballate ed etichettate dal produttore, o in un altro luogo, ma sotto la diretta supervisione del produttore o di un suo rappresentante all'interno della zona geografica definita, in sacchi Raschel da 0,5, 1, 2, 5, 10 e 15\xa0kg o sfuse in casse da 15, 20, 25 e 50\xa0kg o ancora in confezioni da 3, 5 o 10 pezzi. Il prodotto può essere reimballato al di fuori della zona geografica definita.",
 'A questo proposito, il CESE constata che il DCM può essere prodotto, stoccato, trasportato e utilizzato con sicurezza in sistemi chiusi. Il DCM non è infiammabile e non contribuisce alla formazione di ozono a livello del suolo. Nei sistemi aperti, invece, come nel caso della sverniciatura, esso presenta evidenti problemi legati alla volatilità (ossia alla tendenza a evaporare rapidamente), alla densità dei vapori risultanti (si accumula nel punto più basso o dove la ventilazione è inadeguata) e al suo effetto narcotico (provoca perdita di conoscenza e morte). Tutto ciò contribuisce ad aumentare i rischi per i bambini. Il DCM è anche classificato come agente cancerogeno di categoria 3 ed è questo il rischio potenziale che prevale sulle etichette dei prodotti che contengono questa sostanza.',
 "Benché l'allegato delinei una serie di procedure relative a contratti di appalto pre-commerciale che, pur non rientrando — in virtù di clausole di esclusione — nell'ambito di applicazione delle Direttive, sono però conformi al quadro giuridico vigente, esiste pur sempre la possibilità di una violazione, magari inconsapevole, di tale normativa. Il CESE raccomanda dunque ai committenti di esaminare attentamente l'allegato e di seguirne scrupolosamente le raccomandazioni. Nel caso in cui l'amministrazione appaltante o uno dei potenziali fornitori nutra il minimo dubbio, il CESE consiglia vivamente alla prima di ottenere preventivamente da parte della Commissione l'assicurazione che la procedura non viola le norme sugli aiuti di Stato o le clausole di esclusione delle Direttive, e di dare prova di ciò a tutti i potenziali fornitori.",
 'Dato che adesso i materiali hanno maggiore valore rispetto a 5-10 anni fa, molti RAEE sfuggono ai circuiti di raccolta esistenti. La conseguenza è che non sempre essi vengono trattati adeguatamente. Componenti pericolose e senza valore di frigoriferi fuori uso, quali ad esempio i condensatori, vengono eliminate senza essere trattate. Oggi i produttori sono ritenuti responsabili della gestione dei RAEE, su cui hanno scarso controllo, o addirittura nessuno. Tutti gli attori della catena, inclusi quindi i rigattieri e\xa0i commercianti, dovrebbero assumersi le stesse responsabilità.',
 "Il CESE sostiene la proposta esenzione dall'IVA, in quanto tale misura può contribuire in modo significativo a potenziare l'attrazione esercitata dalle ERI conferendo loro al tempo stesso un vantaggio competitivo rispetto a progetti analoghi in altre parti del mondo. Il CESE appoggia perciò l'idea di garantire alle ERI le massime agevolazioni fiscali possibili (compatibilmente con le norme sugli aiuti di Stato). Molte infrastrutture di ricerca esistenti, conformi ai criteri previsti dalla direttiva pertinente per la concessione dello statuto di organizzazioni internazionali, beneficiano già di un regime di esenzione dall'IVA e dalle accise. La procedura attualmente in vigore a questo fine comporta però negoziati lunghi e complessi e causa ritardi nella creazione dell'infrastruttura e notevole incertezza sia giuridica che finanziaria. Un'esenzione automatica come quella prevista dal regolamento eliminerebbe le principali barriere allo sviluppo e al funzionamento delle infrastrutture di ricerca in Europa.",
 "Così come la Commissione, anche il CESE attribuisce grande importanza all'attuazione e all'applicazione delle norme giuridiche vigenti. A questo proposito, e specie nel caso della direttiva relativa al distacco, non basta rivolgere semplici appelli agli Stati membri.: Si deve in particolare anche prestare maggiore importanza all'adozione di provvedimenti efficaci applicabili ai casi transfrontalieri. Il CESE approva inoltre l'invito rivolto dalla Commissione a tutti gli Stati membri affinché diano il buon esempio ratificando e applicando le convenzioni dell'OIL classificate come aggiornate da tale organizzazione.",
 "Nella pianificazione delle TEN-E si dovrebbe assegnare un mandato chiaro all'ENTSO-E e all'ACER nonché definire il ruolo di mediazione dell'UE. Il Libro verde non è sufficientemente esplicito su questo punto. Il CESE si rammarica che la maggior parte dei regolatori europei abbia una missione ufficiale limitata alla creazione di un mercato concorrenziale, senza riferimenti alla sicurezza delle forniture, e che la competenza della Commissione in questo campo non sia definita chiaramente. Esso rileva inoltre che il fatto che i regolatori nazionali siano riuniti in un'Agenzia non ne fa comunque un regolatore europeo. Il CESE si interroga sulla natura giuridica di tale organo, sull'ampiezza dei suoi poteri e sul loro controllo."
]


SPLADE_MODELS = [
    "nickprock/splade-bert-base-italian-xxl-uncased-cv",
    "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
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