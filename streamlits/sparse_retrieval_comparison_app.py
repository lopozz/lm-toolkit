"""
Render precomputed sparse retrieval comparisons.

Expected JSON shape:
{
  "queries": [{
    "qid": "query-id",
    "text": "query text",
    "gold_ids": ["relevant-document-id"],
    "models": {
      "bm25": {
        "ndcg_at_10": 0.75,
        "query_representation": [{"token": "term", "weight": 1.0}],
        "documents": [{
          "rank": 1,
          "doc_id": "document-id",
          "score": 2.3,
          "is_extra_gold": false,
          "text": "document text",
          "expansion": [{"token": "term", "weight": 0.8}]
        }]
      }
    }
  }]
}
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = APP_DIR / "sparse_retrieval_comparison.json"
DISPLAY_DOCUMENTS = 5


def escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


@st.cache_data(show_spinner=False)
def load_queries(path_str: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
        raise ValueError("Expected an object with a 'queries' list.")
    queries = payload["queries"]

    expected_models: set[str] | None = None
    for query in queries:
        if "gold_ids" not in query:
            raise ValueError("Missing 'gold_ids'. Regenerate the JSON with the latest exporter.")
        models = query.get("models")
        if not isinstance(models, dict) or not models:
            raise ValueError("Missing model representations. Regenerate the JSON with the latest exporter.")
        model_names = set(models)
        if expected_models is None:
            expected_models = model_names
        elif model_names != expected_models:
            raise ValueError("Queries contain inconsistent model sets. Regenerate the JSON.")

        for model_name, model_data in models.items():
            if not isinstance(model_data, dict):
                raise ValueError(f"Invalid representation for model '{model_name}'. Regenerate the JSON.")
            if "ndcg_at_10" not in model_data:
                raise ValueError(
                    f"Missing 'ndcg_at_10' for model '{model_name}'. "
                    "Regenerate the JSON with the latest exporter."
                )
            ndcg = float(model_data["ndcg_at_10"])
            if not 0.0 <= ndcg <= 1.0:
                raise ValueError(f"Invalid nDCG@10 for model '{model_name}'. Regenerate the JSON.")
            if "query_representation" not in model_data:
                raise ValueError(
                    f"Missing query representation for model '{model_name}'. "
                    "Regenerate the JSON with the latest exporter."
                )
            documents = model_data.get("documents")
            if not isinstance(documents, list):
                raise ValueError(
                    f"Invalid document list for model '{model_name}'. "
                    "Regenerate the JSON with the latest exporter."
                )
            if any("is_extra_gold" not in document for document in documents):
                raise ValueError(
                    f"Missing gold-document metadata for model '{model_name}'. "
                    "Regenerate the JSON with the latest exporter."
                )
            extra_gold = [document for document in documents if document.get("is_extra_gold", False)]
            top_documents = [document for document in documents if not document.get("is_extra_gold", False)]
            if documents != extra_gold + top_documents or len(top_documents) > DISPLAY_DOCUMENTS:
                raise ValueError(
                    f"Invalid document order for model '{model_name}'. "
                    "Regenerate the JSON with the latest exporter."
                )
            expected_ranks = list(range(1, len(top_documents) + 1))
            if [document.get("rank") for document in top_documents] != expected_ranks:
                raise ValueError(f"Invalid top-document ranks for model '{model_name}'. Regenerate the JSON.")
            query_gold_ids = {str(doc_id) for doc_id in query["gold_ids"]}
            if any(str(document.get("doc_id")) not in query_gold_ids for document in extra_gold):
                raise ValueError(f"Invalid extra gold document for model '{model_name}'. Regenerate the JSON.")
            doc_ids = [str(document.get("doc_id")) for document in documents]
            if len(doc_ids) != len(set(doc_ids)):
                raise ValueError(f"Duplicate documents for model '{model_name}'. Regenerate the JSON.")

    return queries


def sorted_expansion(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (-float(item.get("weight", 0.0)), str(item.get("token", ""))),
    )


def expansion_tokens(items: list[dict[str, Any]]) -> set[str]:
    return {str(item.get("token", "")) for item in items}


def keep_tokens(items: list[dict[str, Any]], tokens: set[str]) -> list[dict[str, Any]]:
    return [item for item in items if str(item.get("token", "")) in tokens]


def token_temperature(weight: float, min_weight: float, max_weight: float, rank: int, total: int) -> float:
    if max_weight > min_weight:
        return (weight - min_weight) / (max_weight - min_weight)
    if total <= 1:
        return 1.0
    return 1.0 - (rank / max(total - 1, 1)) * 0.65


def chip_html(items: list[dict[str, Any]], *, caption: str, limit: int) -> str:
    visible_items = sorted_expansion(items)[:limit]
    if not visible_items:
        return textwrap.dedent(
            f"""
            <div class="panel">
              <div class="panel-title">{escape_html(caption)}</div>
              <div class="empty-state">No active terms.</div>
            </div>
            """
        ).strip()

    weights = [float(item["weight"]) for item in visible_items]
    min_weight = min(weights)
    max_weight = max(weights)
    chips = []

    for rank, item in enumerate(visible_items):
        token = str(item["token"])
        weight = float(item["weight"])
        heat = token_temperature(weight, min_weight, max_weight, rank, len(visible_items))
        hue = 22 + int((1.0 - heat) * 180)
        alpha = 0.18 + heat * 0.42
        border_alpha = 0.3 + heat * 0.5
        font_scale = 0.92 + heat * 0.4
        chips.append(
            textwrap.dedent(
                f"""
                <span
                  class="term-chip"
                  style="
                    background: hsla({hue}, 95%, 55%, {alpha:.3f});
                    border-color: hsla({hue}, 95%, 55%, {border_alpha:.3f});
                    font-size: {font_scale:.2f}rem;
                  "
                >
                  <span class="term-token">{escape_html(token)}</span>
                  <span class="term-weight">{weight:.2f}</span>
                </span>
                """
            ).strip()
        )

    return textwrap.dedent(
        f"""
        <div class="panel">
          <div class="panel-title">{escape_html(caption)}</div>
          <div class="chip-cloud">{''.join(chips)}</div>
        </div>
        """
    ).strip()


def render_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg: #f4efe7;
          --card: rgba(255, 252, 246, 0.84);
          --ink: #1f1c17;
          --muted: #6c645a;
          --line: rgba(31, 28, 23, 0.1);
          --accent: #d95d39;
          --accent-2: #1c7c7d;
        }
        .stApp {
          background:
            radial-gradient(circle at top left, rgba(217, 93, 57, 0.14), transparent 25%),
            radial-gradient(circle at top right, rgba(28, 124, 125, 0.16), transparent 22%),
            linear-gradient(180deg, #f8f4ed 0%, var(--bg) 100%);
          color: var(--ink);
        }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1320px; }
        .hero, .panel, .document-card {
          background: var(--card);
          border: 1px solid var(--line);
          box-shadow: 0 8px 28px rgba(60, 41, 20, 0.05);
        }
        .hero { border-radius: 26px; padding: 1.4rem 1.5rem 1rem; margin-bottom: 1rem; }
        .eyebrow, .panel-title, .document-meta {
          color: var(--muted);
          font-size: 0.78rem;
          font-weight: 700;
          letter-spacing: 0.11em;
          text-transform: uppercase;
        }
        .eyebrow { color: var(--accent-2); }
        .headline { font-size: clamp(1.7rem, 4vw, 2.8rem); margin: 0.3rem 0 0; }
        .query-text { color: var(--ink); font-size: 1.16rem; line-height: 1.5; margin-top: 0.8rem; }
        .panel { border-radius: 20px; padding: 0.95rem; margin: 0.6rem 0 1rem; }
        .panel-title { margin-bottom: 0.75rem; }
        .chip-cloud { display: flex; flex-wrap: wrap; gap: 0.48rem; }
        .term-chip {
          align-items: center;
          border: 1px solid transparent;
          border-radius: 999px;
          display: inline-flex;
          gap: 0.45rem;
          padding: 0.42rem 0.72rem;
        }
        .term-token { font-weight: 700; line-height: 1; }
        .term-weight { font-size: 0.78rem; opacity: 0.72; }
        .document-card { border-radius: 18px; padding: 0.9rem 1rem; margin: 0.7rem 0; }
        .document-card.gold-document {
          background: rgba(28, 124, 125, 0.14);
          border-color: rgba(28, 124, 125, 0.58);
          box-shadow: 0 8px 28px rgba(28, 124, 125, 0.14);
        }
        .document-text { line-height: 1.48; margin-top: 0.45rem; }
        .score-card { margin: 0.2rem 0 0.8rem; }
        .score-label {
          color: var(--muted);
          font-size: 0.86rem;
        }
        .score-value {
          color: var(--ink);
          font-size: 2rem;
          font-weight: 700;
          line-height: 1.2;
        }
        .gold-badge {
          background: var(--accent-2);
          border-radius: 999px;
          color: white;
          display: inline-block;
          font-size: 0.68rem;
          letter-spacing: 0.08em;
          margin-left: 0.45rem;
          padding: 0.16rem 0.42rem;
        }
        .empty-state { color: var(--muted); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_document(
    document: dict[str, Any],
    *,
    gold_ids: set[str],
    query_tokens: set[str] | None,
    top_n: int,
) -> None:
    rank = document.get("rank")
    doc_id = document.get("doc_id", "?")
    score = document.get("score")
    text = str(document.get("text", ""))
    rank_text = str(rank) if rank is not None else "not in saved predictions"
    score_text = f"{float(score):.4f}" if score is not None else "not available"
    is_gold = str(doc_id) in gold_ids
    card_class = "document-card gold-document" if is_gold else "document-card"
    gold_badge = '<span class="gold-badge">Gold</span>' if is_gold else ""
    st.markdown(
        f"""
        <div class="{card_class}">
          <div class="document-meta">Rank {escape_html(rank_text)} · document {escape_html(str(doc_id))} · score {escape_html(score_text)}{gold_badge}</div>
          <div class="document-text">{escape_html(text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    expansion = document.get("expansion", [])
    if query_tokens is not None:
        expansion = keep_tokens(expansion, query_tokens)
    st.markdown(chip_html(expansion, caption="Document representation", limit=top_n), unsafe_allow_html=True)


def render_model(
    model_name: str,
    model_data: dict[str, Any],
    *,
    gold_ids: set[str],
    intersection_only: bool,
    top_n: int,
) -> None:
    st.subheader(model_name)
    st.markdown(
        f"""
        <div class="score-card">
          <div class="score-label">nDCG@10</div>
          <div class="score-value">{float(model_data["ndcg_at_10"]):.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    query_representation = model_data["query_representation"]
    query_tokens = expansion_tokens(query_representation)
    st.markdown("#### Query representation")
    st.markdown(
        chip_html(query_representation, caption="Query token weights", limit=top_n),
        unsafe_allow_html=True,
    )

    documents = model_data["documents"]
    if not documents:
        st.caption("No retrieved documents available.")
        return

    for document in documents:
        render_document(
            document,
            gold_ids=gold_ids,
            query_tokens=query_tokens if intersection_only else None,
            top_n=top_n,
        )


def main() -> None:
    st.set_page_config(
        page_title="Sparse Retrieval Model Inspector",
        page_icon=":material/compare_arrows:",
        layout="wide",
    )
    render_styles()

    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">Sparse Retrieval Diagnostics</div>
          <h1 class="headline">Sparse Retrieval Model Inspector</h1>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Controls")
        data_path = st.text_input("JSON path", value=str(DEFAULT_DATA_PATH))
        top_n = st.slider("Visible terms", min_value=5, max_value=50, value=18)
        search_query = st.text_input("Search query", placeholder="Filter by query text")

    data_file = Path(data_path)
    if not data_file.exists():
        st.error(f"Data file not found: {data_file}")
        return

    try:
        queries = load_queries(str(data_file))
    except (json.JSONDecodeError, ValueError) as exc:
        st.error(f"Could not load comparison data: {exc}")
        return

    if search_query.strip():
        needle = search_query.casefold()
        queries = [query for query in queries if needle in str(query.get("text", "")).casefold()]

    if not queries:
        st.warning("No queries match the current filters.")
        return

    model_names = list(queries[0]["models"])
    left_model_index = model_names.index("bm25") if "bm25" in model_names else 0
    right_model_index = next(
        (index for index, model_name in enumerate(model_names) if index != left_model_index),
        left_model_index,
    )

    with st.sidebar:
        st.subheader("1. Select Models")
        st.caption("Choose which model appears in each comparison column.")
        left_model = st.selectbox("Left column", options=model_names, index=left_model_index)
        right_model = st.selectbox(
            "Right column",
            options=model_names,
            index=right_model_index,
        )
        st.subheader("2. Filter Queries")
        st.caption(
            "Keep only queries whose nDCG@10 falls inside both score ranges. "
            "The sliders filter the query list; they do not change retrieval scores."
        )
        left_range = st.slider(f"Left column: {left_model}", 0.0, 1.0, (0.0, 1.0), step=0.01)
        right_range = st.slider(f"Right column: {right_model}", 0.0, 1.0, (0.0, 1.0), step=0.01)
        intersection_only = st.toggle(
            "Show shared document tokens only",
            help=(
                "Filter each document representation to exact token matches shared with that model's query. "
                "The query representation remains unchanged. "
                "The intersection is computed from all tokens stored in the JSON before the "
                "'Visible terms' limit is applied."
            ),
        )

    def matches_score_filters(query: dict[str, Any]) -> bool:
        models = query["models"]
        left_score = float(models[left_model]["ndcg_at_10"])
        right_score = float(models[right_model]["ndcg_at_10"])
        if not (left_range[0] <= left_score <= left_range[1]):
            return False
        if not (right_range[0] <= right_score <= right_range[1]):
            return False

        return True

    queries = [query for query in queries if matches_score_filters(query)]
    if not queries:
        st.warning("No queries match the current score filters.")
        return

    query_index = st.selectbox(
        "Query",
        options=range(len(queries)),
        format_func=lambda index: f"#{index + 1}  {str(queries[index].get('text', ''))[:100]}",
    )
    query = queries[query_index]

    st.markdown(
        f"""
        <section class="hero">
          <div class="eyebrow">Query #{query_index + 1}</div>
          <div class="query-text">{escape_html(str(query.get("text", "")))}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    models = query["models"]
    gold_ids = {str(doc_id) for doc_id in query["gold_ids"]}
    left_column, right_column = st.columns(2, gap="large")
    with left_column:
        render_model(
            left_model,
            models[left_model],
            gold_ids=gold_ids,
            intersection_only=intersection_only,
            top_n=top_n,
        )

    with right_column:
        render_model(
            right_model,
            models[right_model],
            gold_ids=gold_ids,
            intersection_only=intersection_only,
            top_n=top_n,
        )


if __name__ == "__main__":
    main()
