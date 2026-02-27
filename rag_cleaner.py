import re
from typing import Dict, Optional, List

# ----------------------------
# Regex patterns
# ----------------------------

URL_RE = re.compile(r"https?://\S+")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
BOLD_RE = re.compile(r"\*\*(.*?)\*\*", flags=re.DOTALL)

TOC_LINE_RE = re.compile(r"^\s*\d+(\.\d+)*\s+.*\.{3,}\s*\d+\s*$")

REFERENCE_HEADERS = {
    "references",
    "bibliography",
    "works cited",
    "literature cited",
}

META_PATTERNS = [
    re.compile(r"WantToKnow\.info", re.IGNORECASE),
    re.compile(r"\bPEERS\b"),
    re.compile(r"click here", re.IGNORECASE),
    re.compile(r"subscribe", re.IGNORECASE),
    re.compile(r"all rights reserved", re.IGNORECASE),
]

NAV_LINE_RE = re.compile(r"^\s*(home|about|contact|privacy|terms)\b", re.IGNORECASE)


# ----------------------------
# Core utilities
# ----------------------------

def normalize_newlines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text


def replace_markdown_links(text: str) -> str:
    return MD_LINK_RE.sub(r"\1", text)


def strip_urls(text: str) -> str:
    return URL_RE.sub("", text)


def strip_bold_markers(text: str) -> str:
    return BOLD_RE.sub(r"\1", text)


def remove_navigation_and_meta(text: str) -> str:
    cleaned = []
    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            cleaned.append("")
            continue

        if any(p.search(line) for p in META_PATTERNS):
            continue

        if NAV_LINE_RE.match(stripped):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def remove_toc_blocks(text: str) -> str:
    paragraphs = re.split(r"\n\s*\n", text)
    filtered = []

    for p in paragraphs:
        lines = [l for l in p.splitlines() if l.strip()]
        if len(lines) > 4:
            toc_like = sum(1 for l in lines if TOC_LINE_RE.match(l))
            if toc_like / len(lines) > 0.4:
                continue
        filtered.append(p)

    return "\n\n".join(filtered)


def separate_references(text: str) -> (str, Optional[str]):
    paragraphs = re.split(r"\n\s*\n", text)
    main = []
    refs = []
    in_refs = False

    for p in paragraphs:
        first = p.strip().splitlines()[0].strip().lower() if p.strip() else ""

        if first in REFERENCE_HEADERS:
            in_refs = True

        if in_refs:
            refs.append(p)
        else:
            main.append(p)

    main_text = "\n\n".join(main).strip()
    ref_text = "\n\n".join(refs).strip() if refs else None

    return main_text, ref_text


def cleanup_whitespace(text: str) -> str:
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


# ----------------------------
# RAG Cleaning Entry Point
# ----------------------------

def clean_text_for_rag(
    text: str,
    keep_references: bool = False
) -> str:
    paragraphs = re.split(r"\n\s*\n", text)
    """
    RAG-oriented cleaner:
    - Preserves paragraph structure
    - Removes UI / promo / navigation junk
    - Removes TOC blocks
    - Optionally separates references
    """

    original_len = len(text)

    text = normalize_newlines(text)

    text = replace_markdown_links(text)
    text = strip_bold_markers(text)
    text = strip_urls(text)

    text = remove_navigation_and_meta(text)
    text = remove_toc_blocks(text)

    main_text, references = separate_references(text)
    main_text = cleanup_whitespace(main_text)

    # use the below if we want full-featured info
    #return {
    #    "clean_text": main_text,
    #    "references": references if keep_references else None,
    #    "stats": {
    #        "original_chars": original_len,
    #        "clean_chars": len(main_text),
    #        "retained_pct": round(len(main_text) / original_len * 100, 1)
    #            if original_len else 0.0
    #    }
    #}

    return main_text
