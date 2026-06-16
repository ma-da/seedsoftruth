"""
text_cleaners.py
================

Shared text-cleaning helpers used by the corpus-prep CLIs in this
folder:

  * clean_web_corpus.py   — block-level cleaner for scraped/PDF-derived
                            .txt files (markdown links, bold markers,
                            URL/meta-line stripping, TOC/index/refs
                            detection, paragraph normalization).
  * clean_transcript.py   — drops "Speaker N HH:MM:SS" timestamp lines
                            from podcast/video transcripts.
  * filter_corpus.py      — file-level quality gate: HTML strip,
                            mojibake fix, NFC normalize, sliding-window
                            natural-language detection, garbage rejection.

The helpers here are deliberately importable so the CLIs stay thin and
share one source of truth. Adding a new pattern or heuristic should
almost always happen here, not in a CLI.

Optional dependencies
---------------------
Some helpers require extra packages:

  beautifulsoup4 + lxml   strip_html()
  chardet                  read_text_with_detection()
  ftfy                     normalize_and_clean()

The CLIs that need them import the relevant helper directly and will
raise a clear error at startup if a package is missing.
"""

from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Compiled regex constants
# --------------------------------------------------------------------------- #

URL_RE       = re.compile(r"https?://\S+")
MD_LINK_RE   = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
BOLD_RE      = re.compile(r"\*\*(.*?)\*\*", flags=re.DOTALL)
TOC_LINE_RE  = re.compile(r"^\s*\d+(\.\d+)*\s+.*\.{3,}\s*\d+\s*$")
INLINE_FN_RE = re.compile(r"\[(\d+)\]|\((\d+)\)")
WORD_RE      = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
CTRL_RE      = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
NONPRINTABLE_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

# Cut everything past the first occurrence of this marker. Lifted from
# preprocessing/clean_web_corpus.py — used by WantToKnow-derived corpora
# where the "What you can do" section is non-content boilerplate.
CUT_MARKER = "**What you can do:**"

# Canonical multi-paragraph closers that PEERS/WantToKnow attaches to
# the bottom of nearly every article. Each entry is a *prefix* — the
# cleaner cuts from the first occurrence of any prefix to end of file.
# Order matters only for tie-breaks; the earliest match wins.
DEFAULT_REPEATED_FOOTERS: Tuple[str, ...] = (
    "WantToKnow.info is a nonprofit news information service founded by",
    "Subscribe to our free weekly newsletter",
)

# Substring patterns that flag a file as a JS-required shell rather
# than real content. If any matches AND the file is small after
# cleaning, the file is rejected.
JS_REQUIRED_STUB_PATTERNS: Tuple[str, ...] = (
    "This site requires JavaScript",
    "press the space bar. While dragging",       # Substack a11y prompt
    "Introducing the Substack app",
    "A new economic engine for culture",         # substack.com root
    "Start your Substack",
)

# Regex signals for a Substack-style listing/archive page: a nav strip
# followed by many "<date> • <author>" snippets. Used by is_listing_page.
_SUBSTACK_NAV_RE = re.compile(
    r"Subscribe\s+Sign\s+in\s+Home", re.IGNORECASE
)
_LISTING_DATE_BYLINE_RE = re.compile(
    # Matches  "Mar 13 • Amber Yang"  /  "Dec 26, 2025 • Mark Bailey"
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\s+\d{1,2}(?:,\s*\d{4})?\s*[·•]\s*[A-Z][a-z]+",
)

REFERENCE_HEADERS = {
    "references",
    "bibliography",
    "works cited",
    "literature cited",
}

META_CONTAINS_PATTERNS = [
    re.compile(r"WantToKnow\.info"),
    re.compile(r"\bPEERS\b"),
    re.compile(r"click here", re.IGNORECASE),
]

META_START_PATTERNS = [
    re.compile(r"^\s*note:", re.IGNORECASE),
    re.compile(r"^\s*for more information", re.IGNORECASE),
]

# Transcript speaker + timestamp:
#   "Speaker 1 12:34", "Alice Jones 1:23:45", "Bob 0:05.123"
SPEAKER_LINE_RE = re.compile(
    r"^[A-Za-z]+(?:\s+[A-Za-z]+)*(?:\s+\d+)?\s+"
    r"(?:\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)$"
)


# --------------------------------------------------------------------------- #
# Low-level text helpers
# --------------------------------------------------------------------------- #

def normalize_newlines(text: str) -> str:
    """CRLF/CR -> LF, collapse runs of 4+ blank lines down to 3."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text


def cut_after_marker(text: str, marker: str = CUT_MARKER) -> str:
    """
    Delete everything from the first occurrence of `marker` onward.
    Passing an empty `marker` disables the cut (otherwise `str.find('')`
    would match position 0 and erase the whole document).
    """
    if not marker:
        return text
    idx = text.find(marker)
    if idx == -1:
        return text
    return text[:idx].rstrip() + "\n"


def cut_repeated_footer(
    text: str,
    footer_prefixes: Sequence[str] = DEFAULT_REPEATED_FOOTERS,
) -> str:
    """
    Delete the canonical closer paragraph(s) that PEERS/WTK appends to
    every article. We find the earliest occurrence of any prefix in
    ``footer_prefixes`` and cut from there to end of file.

    Why a separate helper from ``cut_after_marker``: the cut markers
    here are *content-derived* (the boilerplate paragraph itself, not
    an editorial signpost like "What you can do"), so we want to be
    able to register several and cut at the earliest hit.
    """
    if not footer_prefixes:
        return text
    earliest = -1
    for prefix in footer_prefixes:
        if not prefix:
            continue
        idx = text.find(prefix)
        if idx != -1 and (earliest == -1 or idx < earliest):
            earliest = idx
    if earliest == -1:
        return text
    return text[:earliest].rstrip() + "\n"


def replace_markdown_links(text: str) -> str:
    """`[label](https://url)` -> `label`."""
    return MD_LINK_RE.sub(r"\1", text)


def strip_bold_markers(text: str) -> str:
    """`**Something**` -> `Something`."""
    return BOLD_RE.sub(r"\1", text)


def strip_urls(text: str) -> str:
    """Remove bare http(s) URLs. Run AFTER replace_markdown_links so
       linked text isn't lost along with the URL."""
    return URL_RE.sub("", text)


def cleanup_whitespace(text: str) -> str:
    """Trim trailing spaces per line, collapse intraline runs, and
       reduce 3+ blank lines to a single blank line."""
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"^\s+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def remove_meta_lines(text: str) -> str:
    """
    Drop lines that match the META_* pattern lists: promotional
    self-references (WantToKnow, PEERS, "click here") and meta prefixes
    ("Note:", "For more information"). Conservative — only matches
    those specific patterns.
    """
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if any(p.search(line) for p in META_CONTAINS_PATTERNS):
            continue
        if any(p.match(stripped) for p in META_START_PATTERNS):
            continue
        kept.append(line)
    return "\n".join(kept)


def normalize_paragraph_linebreaks(text: str) -> str:
    """
    Collapse wrapped lines inside paragraphs into single-line paragraphs,
    preserving blank-line paragraph boundaries. Useful after PDF→TXT
    conversions that wrap mid-sentence at column boundaries.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n\s*\n", text)
    out = []
    for p in paragraphs:
        single = " ".join(line.strip() for line in p.splitlines())
        single = re.sub(r"\s{2,}", " ", single).strip()
        if single:
            out.append(single)
    return "\n\n".join(out)


# --------------------------------------------------------------------------- #
# Block-based heuristics (TOC / index / refs / footnotes)
# --------------------------------------------------------------------------- #

def split_blocks(text: str, min_block_chars: int = 80) -> List[str]:
    """Paragraph-ish blocks separated by blank lines; drops blocks
       shorter than `min_block_chars`."""
    raw = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    return [b for b in raw if len(b) >= min_block_chars]


def validate_strip_param(value: int, name: str) -> int:
    """Bound strip_pre / strip_post percentages to [0, 25]."""
    if not (0 <= value <= 25):
        raise ValueError(f"{name} must be between 0 and 25 (inclusive), got {value}")
    return value


def strip_blocks(
    blocks: List[str],
    strip_pre: int = 0,
    strip_post: int = 0,
) -> List[str]:
    """Drop a percentage of blocks from the start and end. Useful for
       stripping front-matter / back-matter without page numbers."""
    n = len(blocks)
    if n == 0:
        return blocks
    pre_n = int(n * (strip_pre / 100.0))
    post_n = int(n * (strip_post / 100.0))
    pre_n = min(pre_n, n)
    post_n = min(post_n, n - pre_n)
    return blocks[pre_n : n - post_n]


def is_toc_block(block: str, line_ratio: float = 0.4) -> bool:
    """Heuristic: a block is TOC-like if ≥40% of its lines match
       the `1.2.3  Some Heading .......... 42` pattern."""
    lines = [l for l in block.splitlines() if l.strip()]
    if len(lines) < 5:
        return False
    toc_like = sum(1 for l in lines if TOC_LINE_RE.match(l))
    return (toc_like / len(lines)) >= line_ratio


def is_index_block(block: str) -> bool:
    """
    Heuristic: looks like a book index — mostly short lines that end
    with page numbers and span many initial letters of the alphabet.
    """
    lines = [l.strip() for l in block.splitlines() if l.strip()]
    if len(lines) < 10:
        return False
    if sum(len(l) < 60 for l in lines) / len(lines) < 0.7:
        return False
    if sum(bool(re.search(r"\d+$", l)) for l in lines) / len(lines) < 0.6:
        return False
    initials = {l[0].lower() for l in lines if l and l[0].isalpha()}
    return len(initials) >= 6


def is_footnote_block(block: str) -> bool:
    """A short block where every line starts with `(n)` or `n` — a
       PDF-style footnote stash."""
    lines = block.splitlines()
    if len(lines) > 5:
        return False
    return bool(lines) and all(re.match(r"^\s*\(?\d+\)?\s+.+", l) for l in lines)


def split_reference_blocks(blocks: List[str]) -> Tuple[List[str], List[str]]:
    """
    Once a block's first line equals one of REFERENCE_HEADERS, all
    subsequent blocks are treated as references. Returns
    (main_blocks, ref_blocks).
    """
    main, refs = [], []
    in_refs = False
    for b in blocks:
        first = (b.splitlines()[0].strip().lower() if b.splitlines() else "")
        if first in REFERENCE_HEADERS:
            in_refs = True
        (refs if in_refs else main).append(b)
    return main, refs


def normalize_blocks(blocks: List[str], min_chars: int = 220) -> List[str]:
    """
    Drop blocks shorter than `min_chars` (likely headings) and
    SCREAMING-ALL-CAPS blocks (likely page banners). Conservative.
    """
    out = []
    for b in blocks:
        if len(b) < min_chars:
            continue
        if b.isupper():
            continue
        out.append(b)
    return out


def inline_footnotes(text: str, footnotes: dict) -> str:
    """
    Replace `[n]` / `(n)` markers with the corresponding footnote text
    inline as `[FN: ...]`. Markers without a matching footnote entry
    are deleted. `footnotes` is a dict of `str(n) -> footnote text`.
    """
    def _repl(m):
        """Expand one footnote-marker match to ``[FN: ...]``, or delete it if unknown."""
        idx = m.group(1) or m.group(2)
        note = footnotes.get(idx)
        return f" [FN: {note}]" if note else ""
    return INLINE_FN_RE.sub(_repl, text)


# --------------------------------------------------------------------------- #
# Transcript helpers
# --------------------------------------------------------------------------- #

def is_speaker_line(line: str) -> bool:
    """True if `line` is a `"Speaker N HH:MM:SS"`-style header
       (also matches "Name HH:MM:SS" and "Name HH:MM:SS.sss")."""
    s = line.strip()
    if not s:
        return False
    return bool(SPEAKER_LINE_RE.match(s))


def strip_speaker_lines(text: str) -> str:
    """Remove speaker+timestamp lines, keep dialogue, collapse blank runs."""
    kept = [line.rstrip() for line in text.splitlines() if not is_speaker_line(line)]
    joined = "\n".join(kept)
    return re.sub(r"\n{3,}", "\n\n", joined).strip() + "\n"


# --------------------------------------------------------------------------- #
# File-level natural-language detection (for filter_corpus.py)
# --------------------------------------------------------------------------- #

# Defaults match preprocessing/config.py
WORDS_IN_A_ROW_THRESHOLD = 60
ALPHA_TOKEN_MIN_FRACTION = 0.80
MAX_NONASCII_FRACTION    = 0.20


def read_text_with_detection(p: Path) -> str:
    """
    Read a text file, sniffing the encoding with chardet and falling
    back to utf-8 with `errors='replace'`. Requires `chardet`.
    """
    try:
        import chardet  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "read_text_with_detection() requires chardet. Install with: "
            "pip install chardet"
        ) from e

    raw = p.read_bytes()
    guess = chardet.detect(raw) or {}
    enc = guess.get("encoding") or "utf-8"
    try:
        return raw.decode(enc, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def strip_html(text: str, ext: str = "") -> str:
    """
    If the text looks like HTML (or `ext` is `.html`/`.htm`), extract
    visible text via BeautifulSoup. Requires `beautifulsoup4` and
    ideally `lxml`. Falls back to `html.parser` if lxml is missing.
    """
    looks_like_html = (
        ("<html" in text[:1000].lower())
        or ("</p>" in text.lower())
        or ("<body" in text.lower())
    )
    if ext in {".html", ".htm"} or looks_like_html:
        try:
            from bs4 import BeautifulSoup  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "strip_html() requires beautifulsoup4. Install with: "
                "pip install beautifulsoup4 lxml"
            ) from e
        try:
            soup = BeautifulSoup(text, "lxml")
        except Exception:
            soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
    return html.unescape(text)


def normalize_and_clean(text: str) -> str:
    """
    Aggressive normalization for the file-level quality gate:
    fix mojibake with ftfy, NFC-normalize, drop non-printables,
    replace all non-word punctuation with spaces, collapse runs.
    Produces a single-line "tokenizable" string. Requires `ftfy`.
    """
    try:
        from ftfy import fix_text  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "normalize_and_clean() requires ftfy. Install with: pip install ftfy"
        ) from e

    text = fix_text(text)
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch if (ch.isprintable() or ch in "\n\t ") else " " for ch in text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"[_\-+=~^`|\\/<>{}\[\]()*#%$@:;.,!?]{2,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_natural_language_run(
    clean_text: str,
    window_size: int = WORDS_IN_A_ROW_THRESHOLD,
    alpha_min_frac: float = ALPHA_TOKEN_MIN_FRACTION,
    max_nonascii_frac: float = MAX_NONASCII_FRACTION,
) -> bool:
    """
    Slide a window of `window_size` words. Return True if any window
    has ≥ alpha_min_frac alphabetic tokens AND ≤ max_nonascii_frac
    non-ASCII chars per token (averaged). Used to reject files that
    are mostly OCR garbage / metadata dumps.
    """
    tokens = WORD_RE.findall(clean_text)
    if len(tokens) < window_size:
        return False

    is_alpha = [t.isalpha() for t in tokens]
    nonascii_frac = [
        sum(ord(c) > 127 for c in t) / max(1, len(t))
        for t in tokens
    ]

    alpha_count = sum(is_alpha[:window_size])
    nonascii_avg = sum(nonascii_frac[:window_size]) / window_size

    if alpha_count / window_size >= alpha_min_frac and nonascii_avg <= max_nonascii_frac:
        return True

    for i in range(window_size, len(tokens)):
        alpha_count += is_alpha[i] - is_alpha[i - window_size]
        nonascii_avg += (nonascii_frac[i] - nonascii_frac[i - window_size]) / window_size
        if alpha_count / window_size >= alpha_min_frac and nonascii_avg <= max_nonascii_frac:
            return True

    return False


def is_garbled(clean_text: str, min_chars: int = 400, alpha_min_frac: float = 0.55) -> bool:
    """
    Reject if text is too short or has a low letter-to-nonspace ratio.
    A complement to has_natural_language_run — that one looks for any
    good window; this one looks at the document as a whole.
    """
    if len(clean_text) < min_chars:
        return True
    nospace = clean_text.replace(" ", "")
    if not nospace:
        return True
    alpha = sum(c.isalpha() for c in nospace)
    return alpha / len(nospace) < alpha_min_frac


# --------------------------------------------------------------------------- #
# File-level rejection predicates                                              #
# --------------------------------------------------------------------------- #
#
# These return (rejected: bool, reason: Optional[str]) so callers can both
# decide and log. They operate on the *raw* file text (no cleaning needed
# beforehand) and are safe to call before the per-block pipeline runs.

def is_js_required_stub(
    text: str,
    *,
    max_chars: int = 600,
    patterns: Sequence[str] = JS_REQUIRED_STUB_PATTERNS,
) -> Tuple[bool, Optional[str]]:
    """
    A file is a JS-required shell if it (a) contains any of
    ``patterns`` AND (b) is shorter than ``max_chars``. The size gate
    keeps us from misclassifying a real article that happens to mention
    JavaScript in its body.
    """
    if len(text) > max_chars:
        return False, None
    for p in patterns:
        if p and p in text:
            return True, f"js_required_stub:{p!r}"
    return False, None


def is_listing_page(
    text: str,
    *,
    min_byline_hits: int = 5,
) -> Tuple[bool, Optional[str]]:
    """
    A file is a listing/archive/topic page if it carries the Substack
    nav strip AND has many "<date> • <author>" snippets (one per linked
    article). ``min_byline_hits=5`` is empirically separating: real
    articles average 0–1 such snippets in their body; archive pages
    have 12+.
    """
    if not _SUBSTACK_NAV_RE.search(text):
        return False, None
    hits = len(_LISTING_DATE_BYLINE_RE.findall(text))
    if hits >= min_byline_hits:
        return True, f"listing_page:bylines={hits}"
    return False, None


def evaluate_file_rejection(
    text: str,
    *,
    drop_js_stubs: bool = True,
    drop_listing_pages: bool = True,
    min_chars: int = 400,
) -> Tuple[bool, Optional[str]]:
    """
    Combined gate run BEFORE block-level cleaning. Returns
    ``(rejected, reason)``. The ``min_chars`` check fires last so its
    reason is "too_short" (rather than e.g. "js_required_stub" for the
    same file) when both could apply — easier to read in the summary.
    """
    if drop_js_stubs:
        rej, reason = is_js_required_stub(text)
        if rej:
            return True, reason
    if drop_listing_pages:
        rej, reason = is_listing_page(text)
        if rej:
            return True, reason
    if min_chars > 0 and len(text.strip()) < min_chars:
        return True, f"too_short:{len(text.strip())}<{min_chars}"
    return False, None


# --------------------------------------------------------------------------- #
# High-level convenience: full web-corpus pipeline as a single call
# --------------------------------------------------------------------------- #

def clean_text_web_corpus(
    text: str,
    *,
    strip_pre: int = 0,
    strip_post: int = 0,
    cut_marker: str = CUT_MARKER,
    min_block_chars: int = 220,
    repeated_footer_prefixes: Sequence[str] = DEFAULT_REPEATED_FOOTERS,
) -> dict:
    """
    Full pipeline used by clean_web_corpus.py. Returns a dict with:

        clean_text   str         — main content after all filters
        references   Optional[str] — references section if found
        stats        dict        — original_chars, clean_chars,
                                   blocks_before_strip, blocks_after_strip,
                                   strip_pre_pct, strip_post_pct,
                                   cut_marker_found, repeated_footer_cut

    Order of operations:
        normalize_newlines
        cut_after_marker
        cut_repeated_footer            (NEW — drops PEERS/WTK closer)
        replace_markdown_links / strip_bold_markers / strip_urls
        remove_meta_lines
        cleanup_whitespace
        normalize_paragraph_linebreaks
        split_blocks → strip_blocks → drop is_toc/is_index
        split_reference_blocks → normalize_blocks

    To disable the repeated-footer cut pass an empty sequence.
    """
    original_chars = len(text)
    marker_present = cut_marker in text
    pre_footer_chars = len(text)

    text = normalize_newlines(text)
    text = cut_after_marker(text, marker=cut_marker)
    text = cut_repeated_footer(text, footer_prefixes=repeated_footer_prefixes)
    footer_cut_chars = pre_footer_chars - len(text)
    text = replace_markdown_links(text)
    text = strip_bold_markers(text)
    text = strip_urls(text)
    text = remove_meta_lines(text)
    text = cleanup_whitespace(text)
    text = normalize_paragraph_linebreaks(text)

    blocks = split_blocks(text)
    blocks_before = len(blocks)
    blocks = strip_blocks(blocks, strip_pre=strip_pre, strip_post=strip_post)
    blocks_after = len(blocks)
    blocks = [b for b in blocks if not is_toc_block(b) and not is_index_block(b)]

    main_blocks, ref_blocks = split_reference_blocks(blocks)
    main_blocks = normalize_blocks(main_blocks, min_chars=min_block_chars)

    cleaned = "\n\n".join(main_blocks).strip() + ("\n" if main_blocks else "")
    references = ("\n\n".join(ref_blocks).strip() + "\n") if ref_blocks else None

    return {
        "clean_text":  cleaned,
        "references":  references,
        "stats": {
            "original_chars":      original_chars,
            "clean_chars":         len(cleaned),
            "blocks_before_strip": blocks_before,
            "blocks_after_strip":  blocks_after,
            "strip_pre_pct":       strip_pre,
            "strip_post_pct":      strip_post,
            "cut_marker_found":    marker_present,
            "repeated_footer_cut": footer_cut_chars,
        },
    }
