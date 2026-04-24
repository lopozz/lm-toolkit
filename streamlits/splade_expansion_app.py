from __future__ import annotations

import json
import math
import re
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = APP_DIR / "splade_expansions.json"
DEFAULT_TOKENIZER_MODEL = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
STOP_WORDS = {
    "a",
    "al",
    "alla",
    "con",
    "da",
    "dal",
    "dalla",
    "dei",
    "del",
    "della",
    "di",
    "e",
    "gli",
    "ha",
    "ho",
    "i",
    "il",
    "in",
    "l",
    "la",
    "le",
    "lo",
    "nel",
    "non",
    "per",
    "si",
    "un",
    "una",
}


def escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def preserve_whitespace(text: str) -> str:
    return (
        escape_html(text)
        .replace(" ", "&nbsp;")
        .replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
        .replace("\n", "<br/>")
    )


@st.cache_data(show_spinner=False)
def load_model_rows(path_str: str) -> dict[str, list[dict[str, Any]]]:
    path = Path(path_str)
    payload = json.loads(path.read_text(encoding="utf-8"))
    model_rows: dict[str, list[dict[str, Any]]] = {}

    for model_name, rows in payload.items():
        normalized_rows: list[dict[str, Any]] = []
        for row in rows:
            normalized_row = dict(row)
            normalized_row["expansion"] = sorted(
                normalized_row.get("expansion", []),
                key=lambda item: (-float(item.get("weight", 0.0)), str(item.get("token", ""))),
            )
            normalized_rows.append(normalized_row)
        model_rows[str(model_name)] = normalized_rows

    return model_rows


def normalize_wordpiece(token: str) -> str:
    return token[2:] if token.startswith("##") else token


@st.cache_resource(show_spinner=False)
def load_tokenizer(model_name_or_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


def tokenizer_for_model(model_name: str) -> str:
    known_paths = {
        "opensearch-neural-sparse-encoding-multilingual-v1": "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
        "splade-bert-base-italian-xxl-uncased-cv": "nickprock/splade-bert-base-italian-xxl-uncased-cv",
    }
    return known_paths.get(model_name, model_name)


def original_terms(text: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_RE.finditer(text)}


def classify_terms(text: str, expansion: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_terms = original_terms(text)
    in_text: list[dict[str, Any]] = []
    added: list[dict[str, Any]] = []

    for item in expansion:
        base_token = normalize_wordpiece(str(item["token"])).lower()
        if not base_token or base_token == "[unk]":
            added.append(item)
        elif base_token in source_terms:
            in_text.append(item)
        else:
            added.append(item)

    return in_text, added


def token_temperature(weight: float, min_weight: float, max_weight: float, rank: int, total: int) -> float:
    if max_weight > min_weight:
        return (weight - min_weight) / (max_weight - min_weight)
    if total <= 1:
        return 1.0
    return 1.0 - (rank / max(total - 1, 1)) * 0.65


def chip_html(items: list[dict[str, Any]], *, caption: str, limit: int | None = None) -> str:
    visible_items = items if limit is None else items[:limit]
    if not visible_items:
        return textwrap.dedent(
            f"""
            <div class="panel">
              <div class="panel-title">{escape_html(caption)}</div>
              <div class="empty-state">No terms available for this view.</div>
            </div>
            """
        ).strip()

    weights = [float(item["weight"]) for item in visible_items]
    min_weight = min(weights)
    max_weight = max(weights)
    chips: list[str] = []

    for rank, item in enumerate(visible_items):
        token = str(item["token"])
        weight = float(item["weight"])
        heat = token_temperature(weight, min_weight, max_weight, rank, len(visible_items))
        hue = 22 + int((1.0 - heat) * 180)
        alpha = 0.18 + heat * 0.42
        border_alpha = 0.3 + heat * 0.5
        font_scale = 0.92 + heat * 0.4
        title = " + ".join(str(part) for part in item.get("parts", [token]))
        chips.append(
            textwrap.dedent(
                f"""
                <span
                  class="term-chip"
                  title="{escape_html(title)}"
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


@st.cache_data(show_spinner=False)
def sentence_html(text: str, tokenizer_name_or_path: str) -> str:
    tokenizer = load_tokenizer(tokenizer_name_or_path)
    try:
        encoding = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except Exception:
        return f'<span class="sentence-token missing-tokenizer">{escape_html(text)}</span>'

    palette = [
        "hsla(18, 88%, 72%, 0.55)",
        "hsla(42, 92%, 74%, 0.55)",
        "hsla(74, 90%, 74%, 0.55)",
        "hsla(108, 82%, 76%, 0.55)",
        "hsla(148, 72%, 78%, 0.55)",
        "hsla(188, 72%, 80%, 0.55)",
        "hsla(215, 78%, 82%, 0.55)",
        "hsla(248, 80%, 84%, 0.55)",
        "hsla(286, 78%, 84%, 0.55)",
        "hsla(332, 82%, 82%, 0.55)",
    ]

    parts: list[str] = []
    cursor = 0
    visible_index = 0
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    for index, (token, offsets) in enumerate(zip(tokens, offsets, strict=False)):
        start, end = offsets
        if start == end:
            continue
        if cursor < start:
            parts.append(preserve_whitespace(text[cursor:start]))

        color = palette[visible_index % len(palette)]
        token_text = text[start:end]
        parts.append(
            f"""
            <span
              class="sentence-token"
              style="background:{color};"
              title="{escape_html(token)}"
              data-token="{escape_html(token)}"
            >{preserve_whitespace(token_text)}</span>
            """
        )
        cursor = end
        visible_index += 1

    if cursor < len(text):
        parts.append(preserve_whitespace(text[cursor:]))

    if not parts:
        return f'<span class="sentence-token missing-tokenizer">{escape_html(text)}</span>'

    return "".join(part.strip() for part in parts)


def table_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rank, item in enumerate(items, start=1):
        rows.append(
            {
                "rank": rank,
                "token": item["token"],
                "weight": round(float(item["weight"]), 4),
            }
        )
    return rows


def dominant_patterns(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_counter: Counter[str] = Counter()
    weight_counter: Counter[str] = Counter()
    total_terms = 0
    stopword_hits = 0
    unk_hits = 0

    for row in rows:
        expansion = row["expansion"]
        total_terms += len(expansion)

        for item in expansion:
            token = str(item["token"])
            raw_counter[token] += 1
            weight_counter[token] += int(math.floor(float(item["weight"]) * 100))
            if normalize_wordpiece(token).lower() in STOP_WORDS:
                stopword_hits += 1
            if token == "[UNK]":
                unk_hits += 1

    return {
        "rows": len(rows),
        "avg_terms": total_terms / len(rows) if rows else 0.0,
        "stopword_ratio": stopword_hits / total_terms if total_terms else 0.0,
        "unk_hits": unk_hits,
        "top_raw": raw_counter.most_common(8),
        "top_weighted": weight_counter.most_common(8),
    }


def render_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg: #f4efe7;
          --card: rgba(255, 252, 246, 0.8);
          --ink: #1f1c17;
          --muted: #6c645a;
          --line: rgba(31, 28, 23, 0.08);
          --accent: #d95d39;
          --accent-2: #1c7c7d;
        }
        .stApp {
          background:
            radial-gradient(circle at top left, rgba(217, 93, 57, 0.16), transparent 26%),
            radial-gradient(circle at top right, rgba(28, 124, 125, 0.18), transparent 22%),
            linear-gradient(180deg, #f8f4ed 0%, var(--bg) 100%);
          color: var(--ink);
        }
        .block-container {
          padding-top: 2.2rem;
          padding-bottom: 3rem;
          max-width: 1200px;
        }
        .hero {
          border: 1px solid var(--line);
          background: linear-gradient(135deg, rgba(255,255,255,0.78), rgba(255,245,238,0.88));
          border-radius: 28px;
          padding: 1.6rem 1.6rem 1.2rem;
          box-shadow: 0 18px 50px rgba(60, 41, 20, 0.08);
          margin-bottom: 1rem;
        }
        .eyebrow {
          font-size: 0.8rem;
          letter-spacing: 0.18em;
          text-transform: uppercase;
          color: var(--accent-2);
          font-weight: 700;
          margin-bottom: 0.45rem;
        }
        .headline {
          font-size: clamp(1.8rem, 4vw, 3.2rem);
          line-height: 1;
          margin: 0;
          font-weight: 800;
        }
        .subhead {
          color: var(--muted);
          max-width: 68ch;
          margin-top: 0.8rem;
          font-size: 1rem;
        }
        .sentence-card, .panel, .stat-card {
          background: var(--card);
          border: 1px solid var(--line);
          border-radius: 24px;
          box-shadow: 0 8px 28px rgba(60, 41, 20, 0.05);
        }
        .sentence-card {
          padding: 1.2rem 1.25rem;
          margin: 0.6rem 0 1rem;
        }
        .sentence-label {
          font-size: 0.76rem;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          color: var(--accent);
          font-weight: 700;
        }
        .sentence-text {
          font-size: 1.25rem;
          line-height: 1.55;
          margin-top: 0.45rem;
          white-space: pre-wrap;
        }
        .sentence-token {
          display: inline;
          padding: 0.14rem 0.02rem 0.2rem;
          border-radius: 0.22rem;
          box-decoration-break: clone;
          -webkit-box-decoration-break: clone;
          border-bottom: 2px solid rgba(31, 28, 23, 0.18);
        }
        .missing-tokenizer {
          background: rgba(255, 255, 255, 0.58);
          border: 1px solid rgba(31, 28, 23, 0.12);
        }
        .panel {
          padding: 1rem 1rem 0.95rem;
          margin-bottom: 1rem;
        }
        .panel-title {
          font-size: 0.9rem;
          text-transform: uppercase;
          letter-spacing: 0.11em;
          color: var(--muted);
          font-weight: 700;
          margin-bottom: 0.85rem;
        }
        .chip-cloud {
          display: flex;
          flex-wrap: wrap;
          gap: 0.55rem;
        }
        .term-chip {
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
          border: 1px solid transparent;
          border-radius: 999px;
          padding: 0.48rem 0.8rem;
          color: var(--ink);
        }
        .term-token {
          font-weight: 700;
          line-height: 1;
        }
        .term-weight {
          font-size: 0.8rem;
          opacity: 0.72;
        }
        .stat-card {
          padding: 0.95rem 1rem;
          min-height: 120px;
        }
        .stat-label {
          color: var(--muted);
          font-size: 0.8rem;
          text-transform: uppercase;
          letter-spacing: 0.1em;
          font-weight: 700;
        }
        .stat-value {
          font-size: 2rem;
          font-weight: 800;
          margin-top: 0.45rem;
          line-height: 1;
        }
        .stat-note {
          margin-top: 0.45rem;
          color: var(--muted);
          font-size: 0.92rem;
        }
        .empty-state {
          color: var(--muted);
          font-size: 0.95rem;
        }
        div[data-testid="stDataFrame"] {
          border: 1px solid var(--line);
          border-radius: 18px;
          overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str) -> str:
    return textwrap.dedent(
        f"""
        <div class="stat-card">
          <div class="stat-label">{escape_html(label)}</div>
          <div class="stat-value">{escape_html(value)}</div>
          <div class="stat-note">{escape_html(note)}</div>
        </div>
        """
    ).strip()


def main() -> None:
    st.set_page_config(
        page_title="SPLADE Expansion Inspector",
        page_icon=":material/bubble_chart:",
        layout="wide",
    )
    render_styles()

    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">Sparse Retrieval Diagnostics</div>
          <h1 class="headline">SPLADE Expansion Inspector</h1>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Controls")
        data_path = st.text_input("JSON path", value=str(DEFAULT_DATA_PATH))
        top_n = st.slider("Visible terms", min_value=5, max_value=40, value=18)
        search_query = st.text_input("Search sentence", placeholder="Filter by sentence content")

    data_file = Path(data_path)
    if not data_file.exists():
        st.error(f"Data file not found: {data_file}")
        return

    try:
        model_rows = load_model_rows(str(data_file))
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc.msg}")
        return

    if not model_rows:
        st.warning("The JSON file is empty.")
        return

    model_names = list(model_rows)
    selected_model = st.sidebar.selectbox("Model", options=model_names, index=0)
    tokenizer_model = tokenizer_for_model(selected_model)
    rows = model_rows[selected_model]

    filtered_rows = rows
    if search_query.strip():
        needle = search_query.casefold()
        filtered_rows = [row for row in rows if needle in str(row.get("text", "")).casefold()]

    modes = sorted({str(row.get("mode", "unknown")) for row in filtered_rows}) or ["unknown"]
    selected_mode = st.sidebar.selectbox("Mode", options=modes, index=0)
    filtered_rows = [row for row in filtered_rows if str(row.get("mode", "unknown")) == selected_mode]

    if not filtered_rows:
        st.warning("No rows match the current filters.")
        return

    summaries = dominant_patterns(filtered_rows)
    metric_cols = st.columns(4)
    metric_cols[0].markdown(
        metric_card("Sentences", str(summaries["rows"]), f"{summaries['avg_terms']:.1f} terms on average"),
        unsafe_allow_html=True,
    )
    metric_cols[1].markdown(
        metric_card(
            "Stop-word Ratio",
            f"{summaries['stopword_ratio'] * 100:.0f}%",
            "Terms that look like function words",
        ),
        unsafe_allow_html=True,
    )
    metric_cols[2].markdown(
        metric_card("UNK Activations", str(summaries["unk_hits"]), "Potential tokenizer coverage issues"),
        unsafe_allow_html=True,
    )
    metric_cols[3].markdown(
        metric_card("Model", selected_model, f"{len(filtered_rows)} rows visible"),
        unsafe_allow_html=True,
    )

    sentence_options = {
        f"#{row['id']}  {str(row['text'])[:90]}": row for row in filtered_rows
    }
    selection = st.selectbox("Sentence", options=list(sentence_options))
    selected_row = sentence_options[selection]

    try:
        load_tokenizer(tokenizer_model)
    except Exception as exc:
        st.error(f"Could not load tokenizer '{tokenizer_model}': {exc}")
        return

    st.markdown(
        f"""
        <div class="sentence-card">
          <div class="sentence-label">Sentence #{selected_row['id']} · mode: {escape_html(str(selected_row['mode']))}</div>
          <div class="sentence-text">{sentence_html(str(selected_row['text']), tokenizer_model)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_expansion = selected_row["expansion"]
    in_text, added = classify_terms(str(selected_row["text"]), current_expansion)

    left_col, right_col = st.columns([1.55, 1.0], gap="large")
    with left_col:
        st.markdown(
            chip_html(current_expansion, caption="Tokens ranked by activation", limit=top_n),
            unsafe_allow_html=True,
        )
        st.dataframe(table_rows(current_expansion[:top_n]), width="stretch", hide_index=True)

    with right_col:
        st.markdown(chip_html(in_text, caption="Terms already present in the sentence", limit=top_n), unsafe_allow_html=True)
        st.markdown(chip_html(added, caption="Expansion-only terms / fragments", limit=top_n), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
