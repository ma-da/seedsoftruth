"""
HTML/text extraction helpers for the Seeds of Truth web scraper.

Two strategies are exposed:
- extract_content_newspaper(): newspaper3k-based article extraction.
  Returns (title, text). Best for article pages.
- soup_to_text(): tag-stripping fallback when newspaper3k is not installed
  or fails to extract usable content.

Plus small utilities used elsewhere in the package: clean_url,
hash_html_content, body_adjustments, save_extracted_text_to_file.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Tuple

from bs4 import BeautifulSoup

log = logging.getLogger(__name__)


def clean_url(url: str) -> str:
    """Strip scheme and trailing slash for use as a filesystem-safe key."""
    url = url.replace("https://", "").replace("http://", "")
    return url.rstrip("/")


def hash_html_content(html: bytes | str) -> str:
    """Stable SHA-256 of normalized HTML, used for content de-duplication.

    Accepts either bytes or str. Lowercases and strips before hashing so
    that minor whitespace/case variations do not produce different hashes.
    """
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    normalized = html.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def body_adjustments(soup: BeautifulSoup) -> None:
    """Mutate `soup` in place to remove known noise (e.g., Substack CDN scripts).

    Safe to call before serialization or content extraction.
    """
    body = soup.find("body")
    if body is None:
        return
    for s in body.find_all("script"):
        try:
            if "substackcdn" in (s.get("src") or ""):
                s.decompose()
        except Exception:
            # Defensive: malformed HTML shouldn't crash the crawl.
            continue


def soup_to_text(soup: BeautifulSoup) -> str:
    """Tag-stripping text extraction. Used as a fallback path."""
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    text = soup.get_text(separator=" ")
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return " ".join(chunk for chunk in chunks if chunk)


def extract_content_newspaper(html_content: bytes | str) -> Tuple[str, str]:
    """Extract (title, text) from raw HTML using newspaper3k.

    Returns ("", "") if newspaper3k is unavailable or extraction yields
    nothing. Callers can fall back to `soup_to_text()` on empty result.
    """
    try:
        # Imported lazily so the rest of the package works without newspaper3k.
        from newspaper import Article
    except Exception as e:  # ImportError or transitive failure
        log.debug("newspaper3k unavailable (%s); skipping article extraction", e)
        return "", ""

    try:
        article = Article(url="https://example.invalid")  # dummy; only set_html matters
        article.set_html(html_content)
        article.parse()
        return article.title or "", article.text or ""
    except Exception as e:
        log.warning("newspaper extraction failed: %s", e)
        return "", ""


def save_extracted_text_to_file(txt_filename: str, html_content: bytes | str) -> bool:
    """Extract text from HTML and write it to `txt_filename`.

    Returns True if a non-empty extraction was written, False otherwise.
    Tries newspaper3k first, then falls back to plain tag-stripping so the
    crawler still produces output when newspaper3k is unavailable.
    """
    title, content = extract_content_newspaper(html_content)
    if not content:
        # Fallback: strip tags directly via BeautifulSoup.
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            content = soup_to_text(soup)
            if not title:
                if soup.title and soup.title.string:
                    title = soup.title.string.strip()
        except Exception as e:
            log.warning("fallback soup extraction failed: %s", e)
            return False

    if not content:
        return False

    log.debug("writing extracted text: %s", txt_filename)
    with open(txt_filename, "wb") as f:
        f.write((title.strip() + "\n\n").encode("utf-8"))
        f.write(content.encode("utf-8"))
    return True
