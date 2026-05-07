"""
PDF download and clean-text extraction.

`download_pdf()` fetches a PDF over HTTP. `extract_clean_pdf_text()`
opens it with PyMuPDF (fitz), learns the document's repeated headers
and footers from the first few pages, drops them along with page
numbers and small-font footnote-like blocks, sorts the remaining
blocks into reading order, and reflows the text with light
de-hyphenation.

Returns (title, text). Title comes from PDF metadata when present.
"""

from __future__ import annotations

import logging
import re
import statistics
from collections import Counter
from typing import Optional, Set, Tuple

import requests

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# download                                                                    #
# --------------------------------------------------------------------------- #

def download_pdf(url: str, output_path: str, timeout: int = 30,
                 headers: Optional[dict] = None) -> bool:
    """Download a PDF from `url` to `output_path`. Returns True on success."""
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except Exception as e:
        log.error("PDF download error for %s: %s", url, e)
        return False

    if response.status_code != 200:
        log.error("PDF download failed for %s: status %s", url, response.status_code)
        return False

    with open(output_path, "wb") as f:
        f.write(response.content)
    log.debug("PDF downloaded: %s", output_path)
    return True


def save_text_to_file(title: Optional[str], text: str, output_path: str) -> None:
    """Write extracted text plus title header to a UTF-8 text file."""
    if not title:
        title = "no_title"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n" + text)
    log.debug("PDF text saved: %s", output_path)


# --------------------------------------------------------------------------- #
# extraction helpers                                                          #
# --------------------------------------------------------------------------- #

DIGITS_RE = re.compile(r"^\d{1,4}$")
FOOTNOTE_MARK_RE = re.compile(r"^(\d+|[*†‡])\s")


def _is_probable_page_number(text: str, bbox, page_w: float, page_h: float) -> bool:
    text = text.strip()
    if not DIGITS_RE.match(text):
        return False
    x0, y0, x1, y1 = bbox
    mid_x = (x0 + x1) / 2
    mid_y = (y0 + y1) / 2
    centered = abs(mid_x - page_w / 2) < page_w * 0.2
    top_or_bottom = (mid_y < page_h * 0.08) or (mid_y > page_h * 0.92)
    return centered and top_or_bottom


def _normalize_inline_text(block) -> str:
    parts = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            parts.append(span.get("text", ""))
    return " ".join(" ".join(p.split()) for p in parts)


def _learn_header_footer_bands(pages_dicts, n: int = 5,
                               band_frac: float = 0.10) -> Tuple[Set[str], Set[str]]:
    """Infer repeated header/footer text from the first `n` pages."""
    top_texts: Counter = Counter()
    bot_texts: Counter = Counter()
    for p in pages_dicts[:n]:
        h = p["height"]
        top_y = h * band_frac
        bot_y = h * (1 - band_frac)
        for b in p["blocks"]:
            y0 = b["bbox"][1]
            text = _normalize_inline_text(b).strip()
            if not text:
                continue
            if y0 < top_y:
                top_texts[text] += 1
            elif y0 > bot_y:
                bot_texts[text] += 1

    min_count = max(2, n // 2)
    headers = {t for t, c in top_texts.items() if c >= min_count}
    footers = {t for t, c in bot_texts.items() if c >= min_count}
    return headers, footers


def _page_median_font(pdict) -> float:
    sizes = []
    for b in pdict["blocks"]:
        for ln in b.get("lines", []):
            for sp in ln.get("spans", []):
                if sp.get("size"):
                    sizes.append(sp["size"])
    return statistics.median(sizes) if sizes else 10.0


def _block_avg_font(block) -> float:
    sizes = []
    for ln in block.get("lines", []):
        for sp in ln.get("spans", []):
            if sp.get("size"):
                sizes.append(sp["size"])
    return (sum(sizes) / len(sizes)) if sizes else 0.0


def _should_drop_block(block, page_w: float, page_h: float, med_font: float,
                       headers: Set[str], footers: Set[str]) -> bool:
    text = _normalize_inline_text(block).strip()
    if not text:
        return True
    if _is_probable_page_number(text, block["bbox"], page_w, page_h):
        return True
    if text in headers or text in footers:
        return True
    y0 = block["bbox"][1]
    avg_font = _block_avg_font(block)
    is_bottom_band = y0 > page_h * 0.85
    looks_like_footnote = FOOTNOTE_MARK_RE.match(text) is not None
    much_smaller_font = avg_font and avg_font < 0.8 * med_font
    if is_bottom_band and (much_smaller_font or looks_like_footnote):
        return True
    return False


def _sort_blocks_reading_order(blocks):
    return sorted(blocks, key=lambda b: (round(b["bbox"][1], 1), round(b["bbox"][0], 1)))


def _lines_from_block(block):
    lines = []
    for ln in block.get("lines", []):
        spans = [sp.get("text", "") for sp in ln.get("spans", [])]
        lines.append("".join(spans))
    return lines


def _dehyphenate_and_reflow(lines) -> str:
    out = []
    buf = ""
    for line in lines:
        line = line.rstrip()
        if not line:
            if buf:
                out.append(buf.strip())
                buf = ""
            continue
        if buf:
            if buf.endswith("-"):
                # Common fix: drop hyphen and join if next starts lowercase.
                if line and line[:1].islower():
                    buf = buf[:-1] + line.lstrip()
                else:
                    buf = buf[:-1] + "-" + line.lstrip()
            else:
                if re.match(r"^(\s*[\-•*]|\s*\d+[\.\)])\s", line):
                    out.append(buf.strip())
                    buf = line.strip()
                else:
                    buf += " " + line.strip()
        else:
            buf = line.strip()
    if buf:
        out.append(buf.strip())
    out = [" ".join(p.split()) for p in out]
    return "\n\n".join(out)


# --------------------------------------------------------------------------- #
# public                                                                      #
# --------------------------------------------------------------------------- #

def extract_clean_pdf_text(pdf_path: str, max_pages: Optional[int] = None,
                           learn_pages: int = 5) -> Tuple[Optional[str], str]:
    """Open `pdf_path` and return (title, cleaned_text).

    `max_pages` caps the number of pages processed. `learn_pages` is the
    number of leading pages used to learn repeated header/footer text.
    Raises ImportError if PyMuPDF is not installed.
    """
    import fitz  # PyMuPDF; lazy import so callers without PDFs don't need it

    doc = fitz.open(pdf_path)
    try:
        title = (doc.metadata or {}).get("title")

        pages = []
        for i, page in enumerate(doc):
            if max_pages is not None and i >= max_pages:
                break
            pdict = page.get_text("dict")
            pdict["width"] = page.rect.width
            pdict["height"] = page.rect.height
            pages.append(pdict)

        headers, footers = _learn_header_footer_bands(
            pages, n=min(learn_pages, len(pages))
        )

        all_blocks = []
        for p in pages:
            med_font = _page_median_font(p)
            kept = []
            for b in p["blocks"]:
                if b.get("type", 0) != 0:  # 0 = text; skip images, etc.
                    continue
                if _should_drop_block(b, p["width"], p["height"],
                                      med_font, headers, footers):
                    continue
                kept.append(b)
            all_blocks.extend(_sort_blocks_reading_order(kept))

        lines = []
        for b in all_blocks:
            lines.extend(_lines_from_block(b))
            lines.append("")  # paragraph boundary marker

        text = _dehyphenate_and_reflow(lines)
        return title, text
    finally:
        doc.close()
