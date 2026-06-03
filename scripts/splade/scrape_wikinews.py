#!/usr/bin/env python3
"""
Extract passages from Italian Wikinews without mwxml or mwparserfromhell.

Dependencies:
    pip install requests tqdm

Input:
    itwikinews-latest-pages-articles.xml.bz2

Output:
    JSONL:
    {"_id": "...", "title": "...", "text": "..."}
"""

import argparse
import bz2
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
from tqdm import tqdm


DEFAULT_DUMP_URL = (
    "https://dumps.wikimedia.org/itwikinews/latest/"
    "itwikinews-latest-pages-articles.xml.bz2"
)


BAD_TITLE_PREFIXES = (
    "Wikinotizie:",
    "Discussione:",
    "Discussioni:",
    "Utente:",
    "Discussioni utente:",
    "File:",
    "MediaWiki:",
    "Template:",
    "Aiuto:",
    "Categoria:",
    "Portale:",
    "Modulo:",
    "Speciale:",
)


def download_file(url: str, path: Path, chunk_size: int = 1024 * 1024) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with path.open("wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {path.name}",
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def strip_namespace(tag: str) -> str:
    return tag.split("}", 1)[-1]


def get_child_text(elem: ET.Element, child_name: str) -> str:
    for child in elem:
        if strip_namespace(child.tag) == child_name:
            return child.text or ""
    return ""


def find_child(elem: ET.Element, child_name: str) -> Optional[ET.Element]:
    for child in elem:
        if strip_namespace(child.tag) == child_name:
            return child
    return None


def is_probably_article(title: str, namespace: str) -> bool:
    if namespace != "0":
        return False

    if any(title.startswith(prefix) for prefix in BAD_TITLE_PREFIXES):
        return False

    if title.lower().strip() in {"pagina principale"}:
        return False

    return True


def remove_balanced_templates(text: str) -> str:
    """
    Removes {{...}} blocks with rough nesting support.
    This is not a full MediaWiki parser, but works well enough for many dumps.
    """
    out = []
    i = 0
    depth = 0

    while i < len(text):
        if text[i : i + 2] == "{{":
            depth += 1
            i += 2
            continue

        if depth > 0 and text[i : i + 2] == "}}":
            depth -= 1
            i += 2
            continue

        if depth == 0:
            out.append(text[i])

        i += 1

    return "".join(out)


def clean_wiki_markup(raw: str) -> str:
    text = raw or ""

    # Remove comments.
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)

    # Remove refs.
    text = re.sub(r"<ref[^>/]*/>", " ", text, flags=re.I)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.I | re.S)

    # Remove tables.
    text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.S)

    # Remove templates roughly.
    text = remove_balanced_templates(text)

    # Remove file/category links entirely.
    text = re.sub(
        r"\[\[(File|Immagine|Image|Categoria|Category):[^\]]+\]\]",
        " ",
        text,
        flags=re.I,
    )

    # Convert wiki links:
    # [[target|surface]] -> surface
    text = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", text)

    # [[surface]] -> surface
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    # External links:
    # [https://example.com label] -> label
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)

    # Bare URLs.
    text = re.sub(r"https?://\S+", " ", text)

    # Remove HTML tags.
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove section headings markup but keep heading text.
    text = re.sub(r"={2,}\s*(.*?)\s*={2,}", r"\1", text)

    # Remove bold/italic wiki markup.
    text = text.replace("'''", "")
    text = text.replace("''", "")

    # Remove common leftover bullets.
    text = re.sub(r"^\s*[*#;:]+\s*", "", text, flags=re.M)

    # Drop low-value sections and lines.
    bad_section_names = {
        "fonti",
        "note",
        "voci correlate",
        "collegamenti esterni",
        "altri progetti",
    }

    lines = []
    skip_rest = False

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        normalized = line.lower().strip(" =:")
        if normalized in bad_section_names:
            skip_rest = True
            continue

        if skip_rest:
            continue

        if line.lower().startswith(("categoria:", "file:", "immagine:")):
            continue

        # Skip lines that are mostly punctuation or markup leftovers.
        if len(line) < 20 and re.fullmatch(r"[\W_]+", line):
            continue

        lines.append(line)

    text = " ".join(lines)

    # Cleanup whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_into_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Simple Italian-friendly splitter.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖÈÉÌÒÙ0-9])", text)

    return [s.strip() for s in sentences if s.strip()]


def make_passages(
    text: str,
    min_words: int = 60,
    max_words: int = 180,
    stride_sentences: int = 1,
) -> List[str]:
    sentences = split_into_sentences(text)

    passages = []
    current = []
    current_words = 0

    for sent in sentences:
        n_words = len(sent.split())

        if current and current_words + n_words > max_words:
            passage = " ".join(current).strip()

            if len(passage.split()) >= min_words:
                passages.append(passage)

            if stride_sentences > 0:
                current = current[-stride_sentences:]
                current_words = sum(len(s.split()) for s in current)
            else:
                current = []
                current_words = 0

        current.append(sent)
        current_words += n_words

    if current:
        passage = " ".join(current).strip()
        if len(passage.split()) >= min_words:
            passages.append(passage)

    return passages


def iter_pages_from_bz2_xml(path: Path) -> Iterable[Dict[str, str]]:
    """
    Streaming parser over the compressed Wikimedia XML dump.
    Avoids loading the full dump into memory.
    """
    with bz2.open(path, "rb") as f:
        context = ET.iterparse(f, events=("end",))

        for event, elem in context:
            if strip_namespace(elem.tag) != "page":
                continue

            title = get_child_text(elem, "title")
            namespace = get_child_text(elem, "ns")
            page_id = get_child_text(elem, "id")

            revision = find_child(elem, "revision")
            raw_text = ""

            if revision is not None:
                text_elem = find_child(revision, "text")
                raw_text = text_elem.text or "" if text_elem is not None else ""

            if is_probably_article(title, namespace):
                yield {
                    "page_id": page_id,
                    "title": title,
                    "raw_text": raw_text,
                }

            elem.clear()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dump-url", default=DEFAULT_DUMP_URL)
    parser.add_argument("--download-dir", default="dumps")

    parser.add_argument("--output", required=True)

    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--min-article-words", type=int, default=80)
    parser.add_argument("--min-passage-words", type=int, default=100)
    parser.add_argument("--max-passage-words", type=int, default=800)
    parser.add_argument("--stride-sentences", type=int, default=1)

    args = parser.parse_args()

    if args.download:
        dump_name = args.dump_url.rstrip("/").split("/")[-1]
        dump_path = Path(args.download_dir) / dump_name

        if not dump_path.exists():
            download_file(args.dump_url, dump_path)
    else:
        if not args.dump:
            raise ValueError("Pass --dump or use --download.")
        dump_path = Path(args.dump)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_articles = 0
    n_passages = 0

    with output_path.open("w", encoding="utf-8") as out:
        for page in tqdm(iter_pages_from_bz2_xml(dump_path), desc="Extracting passages"):
            if args.max_articles is not None and n_articles >= args.max_articles:
                break

            title = page["title"]
            clean_text = clean_wiki_markup(page["raw_text"])
            # clean_text = page["raw_text"]

            if len(clean_text.split()) < args.min_article_words:
                continue

            passages = make_passages(
                clean_text,
                min_words=args.min_passage_words,
                max_words=args.max_passage_words,
                stride_sentences=args.stride_sentences,
            )

            if not passages:
                continue

            n_articles += 1

            for passage_idx, passage in enumerate(passages):
                row = {
                    "_id": f"{page['page_id']}_p{passage_idx}",
                    "title": title,
                    "text": passage,
                }

                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_passages += 1

    print("Done.")
    print(f"Articles kept: {n_articles}")
    print(f"Passages written: {n_passages}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()