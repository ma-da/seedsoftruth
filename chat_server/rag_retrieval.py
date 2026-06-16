"""rag_retrieval.py — hybrid SQLite retrieval engine.

Extracted from rag_controller.py (2026-05-20 refactor). Owns retrieval
state and boot, entity-NER normalization, query/text cleaning, scoring
helpers, the five ``_hybrid_search_sqlite`` algorithm variants, the
FTS-branch detector, the min-gate, and the public ``search_references``
dispatcher.

Depends only on the standard library, spaCy and logging_config — it has
no dependency on rag_controller or rag_context, so it can be imported and
tested in isolation.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import string
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import spacy

import logging_config

rag_logger = logging_config.get_logger("rag")

TRINEDAY_TOKEN = "Trine Day Publishing"

# ------------------ CONFIG ------------------

MAX_QUESTION_WORDS = 400
DEFAULT_TOP_K = int(os.getenv("TRINEDAY_TOP_K", "10"))
ENABLE_MIN_GATING = False

# v4 min-gate: top-1 BM25 floor below which queries are declined.
# Calibrated empirically via eval/calibrate_floor.py against the in-domain /
# probe split — at 17.0 we catch 3 of 4 probes (cadmium, schooner, swallow)
# with at most 1/13 in-domain false-decline ("How can we reform government?").
# Switzerland slips through on score alone and is caught by the second condition
# (no canonical entities + broad-OR FTS fallback) inside _min_gate.
MIN_GATE_SCORE_FLOOR = float(os.getenv("MIN_GATE_SCORE_FLOOR", "17.0"))

RETRIEVAL_BACKEND = os.getenv("RETRIEVAL_BACKEND", "sqlite_hybrid")
HYBRID_DB_PATH = Path(
    os.getenv("HYBRID_DB_PATH", "./data/gamma_db_clean.db")
)
ENTITY_CANON_MAP_PATH = Path(
    os.getenv(
        "ENTITY_CANON_MAP_PATH",
        "./data/entity_query_normalization_map.flat.json",
    )
)
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

HYBRID_ENTITY_WEIGHT = float(os.getenv("HYBRID_ENTITY_WEIGHT", "4.0"))
HYBRID_FULLTEXT_WEIGHT = float(os.getenv("HYBRID_FULLTEXT_WEIGHT", "1.0"))
HYBRID_ENTITY_LIMIT = int(os.getenv("HYBRID_ENTITY_LIMIT", "200"))
HYBRID_FULLTEXT_LIMIT = int(os.getenv("HYBRID_FULLTEXT_LIMIT", "200"))

STOPWORDS = {
    "what",
    "about",
    "the",
    "a",
    "an",
    "of",
    "to",
    "for",
    "in",
    "on",
    "at",
    "is",
    "was",
    "are",
    "were",
    "did",
    "does",
    "do",
    "and",
    "or",
    "but",
    "with",
    "from",
    "by",
    "as",
    "it",
    "this",
    "that",
    "these",
    "those",
}

_SPACY_TO_CATEGORY = {
    "PERSON": "persons",
    "ORG": "organizations",
    "GPE": "locations",
    "LOC": "locations",
    "EVENT": "events",
    "WORK_OF_ART": "works",
    "DATE": "dates",
}

_STRIP_PHRASES_RE = re.compile(
    r"(click here|more along these lines|about us)", re.IGNORECASE
)
_STOP_MARKERS = ["</s>", "<|end|>", "<|eot_id|>"]
_URL_RE = re.compile(r"(https?://[^\s<]+)")
_SENT_RE = re.compile(r"[^.!?]+[.!?]*")


# ------------------ STATE ------------------


@dataclass
class RetrievalState:
    """Process-wide retrieval state shared by every search call.

    Attributes:
        backend: Retrieval backend identifier; always ``"sqlite_hybrid"``.
        sqlite_db_path: Path to the hybrid SQLite FTS5 database.
        flat_lookup: Flat ``normalized_key -> canonical_name`` map used to
            normalize spaCy-extracted entity surface forms.
        nlp: Loaded spaCy ``Language`` pipeline used for entity NER and
            query structure analysis.
        _thread_local: ``threading.local`` holder for per-thread SQLite
            connections (see ``_get_sqlite_conn``).
    """

    backend: str
    sqlite_db_path: Path
    flat_lookup: Dict[str, str]
    nlp: Any
    _thread_local: Any


# ------------------ SQLITE / BOOT ------------------


def _get_sqlite_conn(state: RetrievalState) -> sqlite3.Connection:
    """
    One SQLite connection per thread.
    Because sharing one connection across threads is how goblins are born.
    """
    conn = getattr(state._thread_local, "sqlite_conn", None)
    if conn is None:
        # The hybrid corpus DB is read-only at runtime: it is rebuilt offline
        # and the app is restarted afterwards. Open it read-only via the
        # `immutable=1` URI so SQLite never needs write access to the file,
        # its directory, or WAL sidecar (-wal/-shm) files. A non-owner process
        # holding only read permission (0644) can then query it. Without this,
        # SQLite silently downgrades to a read-only handle and the first read
        # against this WAL-mode DB fails with "attempt to write a readonly
        # database" when it tries to create the -shm file.
        db_uri = f"{Path(state.sqlite_db_path).resolve().as_uri()}?immutable=1"
        conn = sqlite3.connect(db_uri, uri=True)
        conn.row_factory = sqlite3.Row
        state._thread_local.sqlite_conn = conn
    return conn


def boot() -> RetrievalState:
    """Initialize and return the shared retrieval state.

    Validates the configured backend, opens the hybrid SQLite database path,
    loads the flat canonical-entity map from JSON, and loads the spaCy model.

    Returns:
        A populated ``RetrievalState`` ready for use by ``search_references``.

    Raises:
        RuntimeError: If ``RETRIEVAL_BACKEND`` is not ``"sqlite_hybrid"``.
        FileNotFoundError: If the hybrid SQLite DB or the flat canonical map
            file does not exist at its configured path.
    """
    rag_logger.info("Booting retrieval...")

    if RETRIEVAL_BACKEND != "sqlite_hybrid":
        raise RuntimeError(
            f"Unsupported RETRIEVAL_BACKEND: {RETRIEVAL_BACKEND}"
        )

    if not HYBRID_DB_PATH.exists():
        raise FileNotFoundError(f"Missing hybrid SQLite DB: {HYBRID_DB_PATH}")

    if not ENTITY_CANON_MAP_PATH.exists():
        raise FileNotFoundError(
            f"Missing flat canonical map: {ENTITY_CANON_MAP_PATH}"
        )

    with ENTITY_CANON_MAP_PATH.open("r", encoding="utf-8") as f:
        flat_lookup = json.load(f)

    nlp = spacy.load(SPACY_MODEL)

    state = RetrievalState(
        backend="sqlite_hybrid",
        sqlite_db_path=HYBRID_DB_PATH,
        flat_lookup=flat_lookup,
        nlp=nlp,
        _thread_local=threading.local(),
    )

    rag_logger.info(f"Boot: loaded hybrid DB from {HYBRID_DB_PATH}")
    rag_logger.info(
        f"Boot: loaded flat lookup entries={len(flat_lookup):,} from {ENTITY_CANON_MAP_PATH}"
    )
    return state


# ------------------ NORMALIZATION / QUERY HELPERS ------------------


def normalize_entity_key(s: str) -> str:
    """Normalize an entity surface form into a lookup key.

    Lowercases, unifies curly quotes, strips possessive ``'s`` suffixes,
    collapses any non-alphanumeric run to a single space, and trims.

    Args:
        s: Raw entity surface text.

    Returns:
        A normalized space-separated key suitable for ``flat_lookup`` lookups.
    """
    s = str(s).strip().lower()
    s = (
        s.replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("‘", "'")
    )
    s = re.sub(r"'s\b", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonical_to_entity_term(s: str) -> str:
    """Convert a canonical entity name into the underscore entity-FTS term.

    Normalizes the name, expands ``&`` to ``and``, and replaces slashes,
    hyphens and spaces with underscores, collapsing repeated underscores.

    Args:
        s: A canonical entity name.

    Returns:
        The underscore-joined entity term as stored in the ``entities_fts``
        index (e.g. ``"john_f_kennedy"``).
    """
    s = normalize_entity_key(s)
    s = s.replace("&", "and")
    s = s.replace("/", "_")
    s = s.replace("-", "_")
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _fts_entity_surface(s: str) -> str:
    """Convert an entity term back to a space-separated surface form.

    Inverse-ish of ``canonical_to_entity_term``: replaces underscores,
    slashes and hyphens with spaces and collapses whitespace. Used to build
    FTS MATCH clauses against the entity index.

    Args:
        s: An entity term (typically underscore-joined).

    Returns:
        The lowercased, space-separated surface string.
    """
    s = str(s).strip().lower()
    s = s.replace("_", " ")
    s = s.replace("/", " ")
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize_fulltext_query(raw_query: str) -> List[str]:
    """Tokenize a raw query into deduplicated full-text search tokens.

    Lowercases, strips non-alphanumeric characters, drops ``STOPWORDS``, and
    removes duplicate tokens while preserving first-seen order.

    Args:
        raw_query: The raw user query string.

    Returns:
        An ordered list of unique, non-stopword query tokens.
    """
    q = str(raw_query).strip().lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    toks = []
    seen = set()
    for t in q.split():
        if not t or t in STOPWORDS:
            continue
        if t not in seen:
            toks.append(t)
            seen.add(t)
    return toks


def build_fulltext_query(raw_query: str, require_all: bool = True) -> str:
    """Build an FTS5 MATCH expression from a raw query.

    Args:
        raw_query: The raw user query string.
        require_all: If True join tokens with ``AND`` (strict); if False join
            with ``OR`` (broad).

    Returns:
        The FTS5 MATCH expression, or an empty string if no tokens remain.
    """
    toks = tokenize_fulltext_query(raw_query)
    if not toks:
        return ""
    joiner = " AND " if require_all else " OR "
    return joiner.join(toks)


def build_entity_match_query(
    entity_terms: List[str], require_all: bool = True
) -> Optional[str]:
    """Build an FTS5 MATCH expression for the entity index.

    Each term is converted to its surface form; multi-word surfaces are
    wrapped in double quotes so FTS5 treats them as exact phrases.

    Args:
        entity_terms: Canonical entity terms to match.
        require_all: If True join terms with ``AND``; if False join with ``OR``.

    Returns:
        The FTS5 MATCH expression, or ``None`` if no usable terms remain.
    """
    cleaned = []
    for term in entity_terms or []:
        t = _fts_entity_surface(term)
        if not t:
            continue
        if " " in t:
            t = f'"{t}"'
        cleaned.append(t)

    if not cleaned:
        return None

    return (" AND " if require_all else " OR ").join(cleaned)


def extract_canonical_entity_terms_typed(
    query: str, state: RetrievalState
) -> List[Tuple[str, str]]:
    """
    Like extract_canonical_entity_terms but preserves the entity category for
    each term. Returns a list of (entity_term, category) pairs. Categories
    come from _SPACY_TO_CATEGORY ("persons" / "organizations" / "locations" /
    "events" / "works" / "dates").

    Used by v4's min-gate to distinguish strong entity signals (persons /
    organizations / events) from weak signals (locations) — a query whose
    only entity match is a country name like "Switzerland" should be treated
    as if it has no entities for gating purposes.
    """
    doc = state.nlp(query)
    seen = set()
    out: List[Tuple[str, str]] = []

    for ent in doc.ents:
        category = _SPACY_TO_CATEGORY.get(ent.label_)
        if not category:
            continue

        raw_text = ent.text.strip()
        if not raw_text:
            continue

        norm_key = normalize_entity_key(raw_text)
        canonical = state.flat_lookup.get(norm_key, norm_key)
        entity_term = canonical_to_entity_term(canonical)

        if entity_term and entity_term not in seen:
            out.append((entity_term, category))
            seen.add(entity_term)

    return out


def extract_canonical_entity_terms(
    query: str, state: RetrievalState
) -> List[str]:
    """Flat-list view of extract_canonical_entity_terms_typed (backward-compat)."""
    return [
        term
        for term, _category in extract_canonical_entity_terms_typed(
            query, state
        )
    ]


# ------------------ RETRIEVAL TEXT CLEANING ------------------


def truncate_question(q: str) -> Tuple[str, bool]:
    """Truncate a question to ``MAX_QUESTION_WORDS`` words.

    Args:
        q: The raw question text.

    Returns:
        A ``(text, truncated)`` tuple where ``text`` is the (possibly
        shortened) question and ``truncated`` is True if words were dropped.
    """
    words = (q or "").split()
    if len(words) <= MAX_QUESTION_WORDS:
        return (q or "").strip(), False
    return " ".join(words[:MAX_QUESTION_WORDS]), True


def _strip_phrases(s: str) -> str:
    """Remove boilerplate web phrases (e.g. "click here") from text.

    Args:
        s: Input text.

    Returns:
        The text with matched boilerplate phrases replaced by spaces.
    """
    if not s:
        return ""
    return _STRIP_PHRASES_RE.sub(" ", str(s))


def _strip_on_literal_stops(
    text: str, stops: Optional[List[str]] = None
) -> str:
    """Truncate text at the first literal stop marker.

    Args:
        text: Input text.
        stops: Stop markers to search for; defaults to ``_STOP_MARKERS``. The
            partial marker ``</s`` is always also matched.

    Returns:
        The text up to (but excluding) the first stop marker, right-stripped;
        the full right-stripped text if no marker is found.
    """
    if not text:
        return ""
    s = str(text)
    stops = list(stops or _STOP_MARKERS)
    escaped = [re.escape(x) for x in stops]
    escaped.append(re.escape("</s"))
    pattern = re.compile(
        r'(?:\s|["\'])*(' + "|".join(escaped) + r')(?:\s|["\'])*',
        flags=re.IGNORECASE,
    )
    m = pattern.search(s)
    if not m:
        return s.rstrip()
    return s[: m.start()].rstrip()


def clean_retrieval_text(text: str) -> str:
    """Normalize raw chunk text for retrieval/context use.

    Truncates at stop markers, strips HTML tags and ``Q:``/``A:`` prefixes,
    removes boilerplate phrases, lowercases, drops control characters, and
    collapses whitespace.

    Args:
        text: Raw chunk or document text.

    Returns:
        The cleaned, lowercased, whitespace-collapsed text.
    """
    if not text:
        return ""
    s = str(text)
    s = _strip_on_literal_stops(s, _STOP_MARKERS)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\b(?:Q:\s*|A:\s*)", " ", s)
    s = _strip_phrases(s)
    s = s.lower().replace("\x00", " ")
    s = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# filter indexes out of results

_PUNCT_CHARS = set(string.punctuation)

_URLISH_RE = re.compile(r"https?://|www\.", re.I)
_BRACKET_CIT_RE = re.compile(r"\[\d{1,3}\]")
_BROKEN_URL_DASH_RE = re.compile(
    r"\b[a-z0-9]+\s*-\s*[a-z0-9]+\s*-\s*[a-z0-9]+", re.I
)
_BIBLIO_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_AUTHOR_QUOTE_RE = re.compile(r'"[^"]{8,}"')
_MULTI_SEMI_RE = re.compile(r";")
_MULTI_SLASH_RE = re.compile(r"/")
_REPEAT_DASH_RE = re.compile(r"(?:\s-\s){3,}")
_REFERENCE_LINE_RE = re.compile(
    r"\[\d{1,3}\].{0,140}?(?:https?://|www\.)", re.I
)


def _punct_stats(text: str) -> dict:
    """Compute punctuation/noise statistics for a piece of text.

    Args:
        text: Input text to analyze.

    Returns:
        A dict with keys: ``length``, ``punct_ratio``, ``dash_ratio``,
        ``max_punct_run``, ``urlish_count``, ``bracket_citation_count``,
        ``broken_url_dash_count``, ``year_count``, ``quote_title_count``,
        ``semicolon_count``, ``slash_count``, ``repeat_dash_runs`` and
        ``reference_line_count``. All counts are zero for empty input.
    """
    s = str(text or "")
    if not s:
        return {
            "length": 0,
            "punct_ratio": 0.0,
            "dash_ratio": 0.0,
            "max_punct_run": 0,
            "urlish_count": 0,
            "bracket_citation_count": 0,
            "broken_url_dash_count": 0,
            "year_count": 0,
            "quote_title_count": 0,
            "semicolon_count": 0,
            "slash_count": 0,
            "repeat_dash_runs": 0,
            "reference_line_count": 0,
        }

    length = len(s)
    punct_count = sum(1 for ch in s if ch in _PUNCT_CHARS)
    dash_count = s.count("-") + s.count("–") + s.count("—")

    runs = re.findall(r"[\-–—.,;:!?/\\|_]{2,}", s)
    max_punct_run = max((len(r) for r in runs), default=0)

    return {
        "length": length,
        "punct_ratio": punct_count / max(length, 1),
        "dash_ratio": dash_count / max(length, 1),
        "max_punct_run": max_punct_run,
        "urlish_count": len(_URLISH_RE.findall(s)),
        "bracket_citation_count": len(_BRACKET_CIT_RE.findall(s)),
        "broken_url_dash_count": len(_BROKEN_URL_DASH_RE.findall(s)),
        "year_count": len(_BIBLIO_YEAR_RE.findall(s)),
        "quote_title_count": len(_AUTHOR_QUOTE_RE.findall(s)),
        "semicolon_count": len(_MULTI_SEMI_RE.findall(s)),
        "slash_count": len(_MULTI_SLASH_RE.findall(s)),
        "repeat_dash_runs": len(_REPEAT_DASH_RE.findall(s)),
        "reference_line_count": len(_REFERENCE_LINE_RE.findall(s)),
    }


def _looks_too_punct_noisy(title: str, text: str) -> bool:
    """
    Filters citation soup / OCR sludge / bibliography dumps.
    Tuned to catch things like:
    [6] Author, "Title," site, 2016, http://...

    Args:
        title: The chunk title.
        text: The chunk body text; only the first 1600 chars are inspected.

    Returns:
        True if the chunk looks like punctuation-heavy citation/OCR noise
        and should be dropped from results.
    """
    title_stats = _punct_stats(title)
    text_stats = _punct_stats(text[:1600])

    # Extremely obvious garbage
    if text_stats["max_punct_run"] >= 8:
        return True

    if text_stats["reference_line_count"] >= 2:
        return True

    if (
        text_stats["bracket_citation_count"] >= 4
        and text_stats["urlish_count"] >= 2
    ):
        return True

    if text_stats["broken_url_dash_count"] >= 3:
        return True

    if text_stats["repeat_dash_runs"] >= 1 and text_stats["urlish_count"] >= 2:
        return True

    # Citation-heavy bibliography sludge
    if (
        text_stats["bracket_citation_count"] >= 3
        and text_stats["year_count"] >= 3
        and text_stats["quote_title_count"] >= 2
    ):
        return True

    # General punctuation overload with URL/citation support
    if text_stats["punct_ratio"] > 0.16 and text_stats["urlish_count"] >= 2:
        return True

    if (
        text_stats["dash_ratio"] > 0.03
        and text_stats["broken_url_dash_count"] >= 2
    ):
        return True

    if text_stats["semicolon_count"] >= 6:
        return True

    if text_stats["slash_count"] >= 8 and text_stats["urlish_count"] >= 1:
        return True

    # Ugly titles are suspicious too
    if title_stats["punct_ratio"] > 0.20 and (
        title_stats["semicolon_count"] >= 2 or title_stats["dash_ratio"] > 0.04
    ):
        return True

    return False


# ------------------ RENDERING / RESULT HELPERS ------------------


def _linkify_plain_urls(html_text: str) -> str:
    """Wrap bare URLs in safe ``<a>`` anchor tags.

    Args:
        html_text: Text that may contain plain ``http(s)://`` URLs.

    Returns:
        The text with URLs replaced by anchors opening in a new tab.
    """
    return _URL_RE.sub(
        r'<a href="\1" target="_blank" rel="noopener noreferrer">\1</a>',
        html_text,
    )


def _truncate_words(text: str, max_words: int) -> str:
    """Truncate text to at most ``max_words`` words.

    Args:
        text: Input text.
        max_words: Maximum number of words to keep.

    Returns:
        The original text if short enough, otherwise the first ``max_words``
        words followed by an ellipsis.
    """
    words = (text or "").split()
    if len(words) <= max_words:
        return text or ""
    return " ".join(words[:max_words]) + "…"


def _snippet_html_from_text(text: str, max_words: int = 600) -> str:
    """Build an HTML-safe snippet from raw text.

    Truncates to ``max_words`` words, HTML-escapes the result, converts
    newlines to ``<br>`` and linkifies plain URLs.

    Args:
        text: Raw snippet text.
        max_words: Maximum word count to retain.

    Returns:
        An HTML-safe snippet string ready for frontend display.
    """
    import html as _html

    t = _truncate_words(text or "", max_words)
    t = _html.escape(t).replace("\n", "<br>")
    return _linkify_plain_urls(t)


def _sentence_split(text: str) -> List[str]:
    """Split text into sentences on terminal punctuation.

    Args:
        text: Input text.

    Returns:
        A list of trimmed, non-empty sentence strings.
    """
    s = re.sub(r"\s+", " ", text or "").strip()
    if not s:
        return []
    return [
        m.group(0).strip() for m in _SENT_RE.finditer(s) if m.group(0).strip()
    ]


def _to_term_set(query: str) -> set[str]:
    """Reduce a query to a set of distinct lowercased terms.

    Strips non-alphanumeric characters and drops tokens of length <= 2.

    Args:
        query: The raw query string.

    Returns:
        A set of distinct query terms longer than two characters.
    """
    q = (query or "").lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    return {w for w in q.split() if len(w) > 2}


# Function words and interrogatives that should not contribute to query/doc
# token-overlap counts. Used by build_context_v2 to avoid the "one-token
# match on a polysemous word" failure mode (e.g., a query like "Who killed
# JFK?" matching JFK-Airport sentences via the lone token "jfk").
_QUERY_STOPWORDS: frozenset = frozenset(
    {
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "which",
        "whom",
        "is",
        "are",
        "was",
        "were",
        "be",
        "being",
        "been",
        "am",
        "do",
        "does",
        "did",
        "doing",
        "done",
        "have",
        "has",
        "had",
        "having",
        "can",
        "could",
        "should",
        "would",
        "will",
        "may",
        "might",
        "must",
        "shall",
        "the",
        "and",
        "but",
        "not",
        "nor",
        "of",
        "for",
        "with",
        "into",
        "from",
        "about",
        "between",
        "among",
        "this",
        "that",
        "these",
        "those",
        "such",
        "they",
        "them",
        "their",
        "theirs",
        "you",
        "your",
        "yours",
        "our",
        "ours",
        "tell",
        "show",
        "explain",
        "describe",
        "give",
        "list",
        "any",
        "all",
        "each",
        "every",
        "both",
        "either",
        "neither",
        "some",
        "than",
        "then",
        "very",
        "much",
        "more",
        "most",
        "less",
    }
)


def _tokenize_simple(t: str) -> List[str]:
    """Tokenize text into lowercased alphanumeric words (no stopword filter).

    Args:
        t: Input text.

    Returns:
        A list of lowercased word tokens, in order, including duplicates.
    """
    t = (t or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return [w for w in t.split() if w]


def _trigrams(tokens: List[str]) -> List[str]:
    """Build the list of consecutive 3-token shingles.

    Args:
        tokens: An ordered list of tokens.

    Returns:
        A list of space-joined trigram strings; empty if fewer than 3 tokens.
    """
    return [" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)]


def _fts_positive_score(bm25_score: float) -> float:
    """Map an FTS5 bm25 score to a bounded positive score in ``(0, 1]``.

    FTS5 bm25 returns negative values (more negative = better). This converts
    the magnitude into a positive score that decreases as relevance worsens.

    Args:
        bm25_score: A raw FTS5 bm25 score.

    Returns:
        ``1 / (1 + |bm25_score|)`` — a positive score in ``(0, 1]``.
    """
    return 1.0 / (1.0 + abs(float(bm25_score)))


def _fts_positive_score_simple(bm25_score: float) -> float:
    """Map an FTS5 bm25 score to a positive score by sign inversion.

    FTS5 bm25 returns negative values (more negative = better), so simple
    negation yields a positive score where larger is better.

    Args:
        bm25_score: A raw FTS5 bm25 score.

    Returns:
        ``-bm25_score`` — the sign-inverted score.
    """
    return -float(bm25_score)


def _sqlite_row_to_result(
    row: sqlite3.Row,
    hybrid_score: float,
    entity_score: float,
    fulltext_score: float,
    raw_score: float,
) -> Dict[str, Any]:
    """Convert a ``chunks`` SQLite row into a result dict.

    Args:
        row: A ``sqlite3.Row`` from the ``chunks`` table.
        hybrid_score: The final (post-bonus) hybrid score for the row.
        entity_score: The entity-branch component score.
        fulltext_score: The full-text-branch component score.
        raw_score: The raw (pre-weighting) FTS branch score.

    Returns:
        A result dict with keys including ``row_id``, ``lookup_id``,
        ``source_url``/``source``, ``title``, ``subset``, ``publisher``,
        ``found_on``, ``text``, ``snippet``, ``snippet_html``,
        ``score_bm25`` (set to ``hybrid_score``), ``entity_score``,
        ``fulltext_score`` and ``raw_score``.
    """
    txt = str(row["fulltext_text"] or "")
    src = str(row["source_url"] or "").strip()
    subset = str(row["subset_name"] or "").strip()

    return {
        "row_id": row["chunk_id"] or row["lookup_id"],
        # Integer PK of the row in `chunks`. Exposed so callers (e.g. V5's
        # topic-boost re-ranker) can join back to chunk_topics / chunk_entities
        # without a second roundtrip. Distinct from row_id, which falls back
        # to the string chunk_id for client-facing display.
        "lookup_id": (
            int(row["lookup_id"]) if row["lookup_id"] is not None else None
        ),
        "source_url": src,
        "source": src,
        "title": row["title"] or "",
        "subset": subset,
        "publisher": "",
        "found_on": "",
        "text": txt,
        "snippet": txt[:1200],
        "snippet_html": _snippet_html_from_text(txt, max_words=600),
        "score_bm25": float(hybrid_score),
        "entity_score": float(entity_score),
        "fulltext_score": float(fulltext_score),
        "raw_score": float(raw_score),
    }


def clean_rag_references(docs):
    """
    Clean references for frontend display only.
    Do not use before context building / RAG.
    """
    cleaned = []

    for doc in docs or []:
        new_doc = dict(doc or {})

        subset = str(new_doc.get("subset") or "").strip()
        publisher = str(new_doc.get("publisher") or "").strip()
        found_on = str(new_doc.get("found_on") or "").strip()

        subset_low = subset.lower()

        is_trineday = (
            subset_low == "trine day"
            or subset_low == "trineday"
            or subset_low.startswith("trine day")
            or subset_low.startswith("trineday")
            or publisher == TRINEDAY_TOKEN
            or found_on == TRINEDAY_TOKEN
        )

        if is_trineday:
            # Force the visible subset label you want
            new_doc["subset"] = "Trine Day"

            # Replace display text only
            protected = (
                "This text is protected by copyright and cannot be displayed."
            )
            new_doc["text"] = protected
            new_doc["snippet"] = protected
            new_doc["snippet_html"] = protected

        cleaned.append(new_doc)

    return cleaned


# ------------------ HYBRID SQLITE SEARCH ------------------


def _entity_positions(entity_term: str, text: str) -> List[int]:
    """Find character offsets where an entity surface form occurs in text.

    Args:
        entity_term: An entity term (underscores treated as spaces).
        text: The text to search (case-insensitive, word-bounded).

    Returns:
        A list of character start offsets of each whole-word match.
    """
    surf = re.escape(entity_term.replace("_", " "))
    return [
        m.start()
        for m in re.finditer(rf"\b{surf}\b", str(text or ""), flags=re.I)
    ]


def _has_entity_proximity_match(
    entity_terms: List[str], text: str, max_chars: int = 700
) -> bool:
    """
    Require at least 2 distinct entity terms to appear near each other.
    This helps block "same giant chunk, unrelated subtopics" junk.

    Args:
        entity_terms: Canonical entity terms to look for.
        text: Chunk text; only the first 2500 chars are scanned.
        max_chars: Maximum character distance for two terms to count as near.

    Returns:
        True if at least two distinct terms occur within ``max_chars`` of
        each other; False otherwise.
    """
    s = str(text or "")[:2500]

    present = []
    for term in entity_terms:
        pos = _entity_positions(term, s)
        if pos:
            present.append((term, pos))

    if len(present) < 2:
        return False

    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            for p1 in present[i][1]:
                for p2 in present[j][1]:
                    if abs(p1 - p2) <= max_chars:
                        return True

    return False


def _has_early_anchor(entity_terms: List[str], title: str, text: str) -> bool:
    """
    At least one entity should show up in the title or early text.
    Prevents deep-citation tangents from floating to the top.

    Args:
        entity_terms: Canonical entity terms to look for.
        title: The chunk title.
        text: The chunk body; only the first 600 chars are checked.

    Returns:
        True if any entity surface form appears in the title or early text.
    """
    title_low = str(title or "").lower()
    early = str(text or "")[:600].lower()

    for term in entity_terms:
        surf = term.replace("_", " ").lower()
        if surf in title_low or surf in early:
            return True
    return False


def _generic_title_penalty(title: str) -> float:
    """
    Small penalty for ultra-generic titles.
    Because 'WTK stops' is not exactly screaming relevance.

    Args:
        title: The chunk title.

    Returns:
        A small non-negative penalty: 0.15 for empty titles, 0.25 for the
        known-bad ``"wtk stops"``, 0.10 for titles of two words or fewer,
        otherwise 0.0.
    """
    t = str(title or "").strip().lower()
    if not t:
        return 0.15
    if t in {"wtk stops"}:
        return 0.25
    if len(t.split()) <= 2:
        return 0.10
    return 0.0


def _tokenize_keywords(raw_query: str) -> List[str]:
    """Tokenize a query into keyword tokens (alias of ``tokenize_fulltext_query``).

    Args:
        raw_query: The raw query string.

    Returns:
        An ordered list of unique, non-stopword query tokens.
    """
    return tokenize_fulltext_query(raw_query)


def _keyword_overlap_bonus(raw_query: str, title: str, text: str) -> float:
    """
    Bonus for literal keyword overlap in title + early text.
    Helps excerpt search and named-term queries.

    Args:
        raw_query: The raw query string.
        title: The chunk title.
        text: The chunk body; only the first 3000 chars are checked.

    Returns:
        A non-negative bonus of ``0.22`` per query token found in the title
        plus ``0.05`` per query token found in the early text.
    """
    q_toks = _tokenize_keywords(raw_query)
    if not q_toks:
        return 0.0

    title_low = str(title or "").lower()
    text_low = str(text or "")[:3000].lower()

    hits_title = 0
    hits_text = 0

    for tok in q_toks:
        if tok in title_low:
            hits_title += 1
        if tok in text_low:
            hits_text += 1

    return (0.22 * hits_title) + (0.05 * hits_text)


def _exact_phrase_bonus(raw_query: str, title: str, text: str) -> float:
    """
    Bonus for exact phrase matches and long subphrase matches.
    This is the secret sauce for pasted excerpts.

    Args:
        raw_query: The raw query string; queries shorter than 20 chars
            receive no bonus.
        title: The chunk title.
        text: The chunk body; only the first 5000 chars are checked.

    Returns:
        A non-negative bonus: ``1.25``/``1.0`` for a full-phrase match in the
        title/text, ``0.8``/``0.55`` for a long (5-10 word) subphrase match in
        the title/text, otherwise ``0.0``.
    """
    q = str(raw_query or "").strip().lower()
    q = re.sub(r"\s+", " ", q)

    if len(q) < 20:
        return 0.0

    title_low = str(title or "").lower()
    text_low = str(text or "")[:5000].lower()

    # Full exact phrase
    if q in title_low:
        return 1.25
    if q in text_low:
        return 1.0

    # Subphrase fallback
    words = q.split()
    for span in (10, 8, 7, 6, 5):
        if len(words) >= span:
            for i in range(len(words) - span + 1):
                phrase = " ".join(words[i : i + span])
                if phrase in title_low:
                    return 0.8
                if phrase in text_low:
                    return 0.55

    return 0.0


def _hybrid_search_sqlite(
    state: RetrievalState,
    *,
    entity_terms: List[str],
    fulltext_query: str,
    top_k: int,
    require_all_entities: bool = True,
    subsets: List[str] = None,
) -> List[Dict[str, Any]]:
    """Run the v1 hybrid retrieval algorithm.

    Runs an entity-FTS branch and a full-text-FTS branch (full-text uses a
    strict-AND then broad-OR fallback ladder; the phrase branch is disabled
    here), merges their sign-inverted bm25 scores per chunk, and ranks by a
    weighted sum (``HYBRID_ENTITY_WEIGHT`` / ``HYBRID_FULLTEXT_WEIGHT``).
    A prefiltered candidate pool is then hard-filtered by punctuation noise,
    entity proximity (when 2+ entities), and early-anchor checks, and rescored
    with keyword-overlap and exact-phrase bonuses minus a generic-title penalty.

    Args:
        state: Shared retrieval state.
        entity_terms: Canonical entity terms for the entity branch.
        fulltext_query: Raw query text for the full-text branch.
        top_k: Number of results to return.
        require_all_entities: If True the entity MATCH joins terms with AND.
        subsets: Optional list of subset names to restrict results to.

    Returns:
        A list of up to ``top_k`` result dicts (see ``_sqlite_row_to_result``)
        each augmented with debug keys (``hybrid_score``, ``keyword_bonus``,
        ``phrase_bonus``, ``search_closeness``, ``entity_query_used``,
        ``fulltext_query_used``, ...); ``[]`` if nothing matched.
    """
    conn = _get_sqlite_conn(state)
    cur = conn.cursor()

    entity_query = build_entity_match_query(
        entity_terms,
        require_all=require_all_entities,
    )

    raw_ft_query = str(fulltext_query or "").strip()
    phrase_fulltext_query = ""
    strict_fulltext_query = build_fulltext_query(raw_ft_query, require_all=True)
    broad_fulltext_query = build_fulltext_query(raw_ft_query, require_all=False)

    rag_logger.info(f"raw_ft_query: {raw_ft_query}")
    rag_logger.info(f"phrase_fulltext_query: {phrase_fulltext_query}")
    rag_logger.info(f"strict_fulltext_query: {strict_fulltext_query}")
    rag_logger.info(f"broad_fulltext_query: {broad_fulltext_query}")

    # ----------------------------
    # 1) Entity search
    # ----------------------------
    entity_rows = []
    if entity_query:
        entity_rows = list(
            cur.execute(
                """
                SELECT rowid AS lookup_id, bm25(entities_fts) AS bm25_score
                FROM entities_fts
                WHERE entities_fts MATCH ?
                ORDER BY bm25_score
                LIMIT ?
                """,
                (entity_query, HYBRID_ENTITY_LIMIT),
            )
        )

    # ----------------------------
    # 2) Fulltext search
    # Search order:
    #   a) exact phrase for long excerpt-like queries
    #   b) strict AND
    #   c) broad OR
    # ----------------------------
    fulltext_rows = []
    fulltext_query_used = ""

    if phrase_fulltext_query:
        fulltext_rows = list(
            cur.execute(
                """
                SELECT rowid AS lookup_id, bm25(fulltext_fts) AS bm25_score
                FROM fulltext_fts
                WHERE fulltext_fts MATCH ?
                ORDER BY bm25_score
                LIMIT ?
                """,
                (phrase_fulltext_query, HYBRID_FULLTEXT_LIMIT),
            )
        )
        if fulltext_rows:
            fulltext_query_used = phrase_fulltext_query

    if not fulltext_rows and strict_fulltext_query:
        fulltext_rows = list(
            cur.execute(
                """
                SELECT rowid AS lookup_id, bm25(fulltext_fts) AS bm25_score
                FROM fulltext_fts
                WHERE fulltext_fts MATCH ?
                ORDER BY bm25_score
                LIMIT ?
                """,
                (strict_fulltext_query, HYBRID_FULLTEXT_LIMIT),
            )
        )
        if fulltext_rows:
            fulltext_query_used = strict_fulltext_query

    if (
        not fulltext_rows
        and broad_fulltext_query
        and broad_fulltext_query != strict_fulltext_query
    ):
        fulltext_rows = list(
            cur.execute(
                """
                SELECT rowid AS lookup_id, bm25(fulltext_fts) AS bm25_score
                FROM fulltext_fts
                WHERE fulltext_fts MATCH ?
                ORDER BY bm25_score
                LIMIT ?
                """,
                (broad_fulltext_query, HYBRID_FULLTEXT_LIMIT),
            )
        )
        if fulltext_rows:
            fulltext_query_used = broad_fulltext_query

    # ----------------------------
    # 3) Merge branch scores
    # ----------------------------
    merged: Dict[int, Dict[str, float]] = {}

    for row in entity_rows:
        lookup_id = int(row["lookup_id"])
        raw_score = _fts_positive_score_simple(row["bm25_score"])
        score = _fts_positive_score_simple(row["bm25_score"])
        # score = _fts_positive_score(row["bm25_score"])
        merged.setdefault(
            lookup_id,
            {
                "entity_score": 0.0,
                "fulltext_score": 0.0,
                "raw_score": 0.0,
            },
        )
        merged[lookup_id]["entity_score"] = max(
            merged[lookup_id]["entity_score"],
            score,
        )
        merged[lookup_id]["raw_score"] = max(
            merged[lookup_id]["raw_score"],
            raw_score,
        )

    for row in fulltext_rows:
        lookup_id = int(row["lookup_id"])
        raw_score = _fts_positive_score_simple(row["bm25_score"])
        score = _fts_positive_score_simple(row["bm25_score"])
        # score = _fts_positive_score(row["bm25_score"])
        merged.setdefault(
            lookup_id,
            {
                "entity_score": 0.0,
                "fulltext_score": 0.0,
                "raw_score": 0.0,
            },
        )
        merged[lookup_id]["fulltext_score"] = max(
            merged[lookup_id]["fulltext_score"],
            score,
        )
        merged[lookup_id]["raw_score"] = max(
            merged[lookup_id]["raw_score"],
            raw_score,
        )

    # ----------------------------
    # 4) Initial ranking
    # ----------------------------
    ranked = []
    for lookup_id, parts in merged.items():
        hybrid_score = (
            HYBRID_ENTITY_WEIGHT * parts["entity_score"]
            + HYBRID_FULLTEXT_WEIGHT * parts["fulltext_score"]
        )

        ranked.append(
            (
                lookup_id,
                hybrid_score,
                parts["entity_score"],
                parts["fulltext_score"],
                parts["raw_score"],
            )
        )

    ranked.sort(key=lambda x: x[1], reverse=True)

    # Pull a bigger pool before filtering/bonuses.
    # Otherwise the good excerpt matches may never get their turn on stage.
    prefilter_limit = max(top_k * 20, 200)
    prefilter_limit = min(prefilter_limit, 2000)
    ranked = ranked[:prefilter_limit]

    if not ranked:
        rag_logger.info(
            "Hybrid search empty. entity_terms=%s entity_query=%s fulltext_query_used=%s entity_hits=%d fulltext_hits=%d",
            entity_terms,
            entity_query,
            fulltext_query_used,
            len(entity_rows),
            len(fulltext_rows),
        )
        return []

    ids = [r[0] for r in ranked]
    placeholders = ",".join("?" for _ in ids)

    query = f"""
        SELECT lookup_id, chunk_id, title, subset_name, domain, fulltext_text, source_url
        FROM chunks
        WHERE lookup_id IN ({placeholders})
    """

    if not ids:
        return [{}]
    params = list(ids)

    # --- optional subset filter ---
    if subsets and len(subsets) > 0:
        subset_placeholders = ",".join("?" for _ in subsets)
        query += f" AND subset_name IN ({subset_placeholders})"
        params.extend(subsets)
        rag_logger.info(f"Subsets used in hybrid_search: {subsets}")
    else:
        rag_logger.info("No Subsets used in hybrid_search")

    meta_rows = list(cur.execute(query, params))
    meta_map = {int(r["lookup_id"]): r for r in meta_rows}

    # ----------------------------
    # 5) Post-filter + score bonuses
    # ----------------------------
    rescored = []
    dropped_noisy = 0
    dropped_proximity = 0
    dropped_anchor = 0

    for (
        lookup_id,
        hybrid_score,
        entity_score,
        fulltext_score,
        raw_score,
    ) in ranked:
        row = meta_map.get(lookup_id)
        if row is None:
            continue

        title = str(row["title"] or "")
        fulltext_text = str(row["fulltext_text"] or "")

        if _looks_too_punct_noisy(title, fulltext_text):
            dropped_noisy += 1
            continue

        if len(entity_terms) >= 2:
            if not _has_entity_proximity_match(
                entity_terms, fulltext_text, max_chars=700
            ):
                dropped_proximity += 1
                continue

        if entity_terms:
            if not _has_early_anchor(entity_terms, title, fulltext_text):
                dropped_anchor += 1
                continue

        keyword_bonus = _keyword_overlap_bonus(
            raw_ft_query, title, fulltext_text
        )
        phrase_bonus = _exact_phrase_bonus(raw_ft_query, title, fulltext_text)
        title_penalty = _generic_title_penalty(title)

        adjusted_hybrid_score = (
            hybrid_score - title_penalty + keyword_bonus + phrase_bonus
        )

        rescored.append(
            (
                lookup_id,
                adjusted_hybrid_score,
                entity_score,
                fulltext_score,
                keyword_bonus,
                phrase_bonus,
                raw_score,
            )
        )

    rescored.sort(key=lambda x: x[1], reverse=True)
    rescored = rescored[:top_k]

    results = []
    for (
        lookup_id,
        adjusted_hybrid_score,
        entity_score,
        fulltext_score,
        keyword_bonus,
        phrase_bonus,
        raw_score,
    ) in rescored:
        row = meta_map.get(lookup_id)
        if row is None:
            continue

        result = _sqlite_row_to_result(
            row,
            adjusted_hybrid_score,
            entity_score,
            fulltext_score,
            raw_score,
        )

        # Debug goodies for console / tuning
        result["hybrid_score"] = float(adjusted_hybrid_score)
        result["entity_score"] = float(entity_score)
        result["fulltext_score"] = float(fulltext_score)
        result["keyword_bonus"] = float(keyword_bonus)
        result["phrase_bonus"] = float(phrase_bonus)
        result["search_closeness"] = float(adjusted_hybrid_score)
        result["entity_query_used"] = entity_query or ""
        result["fulltext_query_used"] = fulltext_query_used or ""
        result["subset"] = row["subset_name"] or ""
        result["raw_score"] = float(raw_score)

        results.append(result)

    rag_logger.info(
        "Hybrid search final. entity_terms=%s entity_query=%s fulltext_query_used=%s entity_hits=%d fulltext_hits=%d kept=%d dropped_noisy=%d dropped_proximity=%d dropped_anchor=%d",
        entity_terms,
        entity_query,
        fulltext_query_used,
        len(entity_rows),
        len(fulltext_rows),
        len(results),
        dropped_noisy,
        dropped_proximity,
        dropped_anchor,
    )

    return results


#
# ----------------------------
# Tunables (AB-test these)
# ----------------------------
#

HYBRID_ENTITY_WEIGHT_V2 = HYBRID_ENTITY_WEIGHT * 0.75

ENTITY_COVERAGE_WEIGHT = 0.55
ENTITY_RARITY_WEIGHT = 0.30

COOCCUR_BONUS = 0.15

PROX_BONUS_NEAR = 0.15
PROX_BONUS_FAR = 0.05

NO_ANCHOR_PENALTY = 0.25
IDF_CAP = 3.0


ENTITY_TYPE_WEIGHT = {
    "person": 1.0,
    "event": 1.0,
    "organization": 0.7,
    "work": 0.5,
    "location": 0.4,
    "date": 0.2,
}


def _hybrid_search_sqlite2(
    state,
    *,
    entity_terms,
    fulltext_query,
    top_k,
    require_all_entities=True,
    subsets=None,
) -> List[Dict[str, Any]]:
    """Run the v2 hybrid retrieval algorithm (structured-entity rescoring).

    Like v1 but: (a) enables the exact-phrase FTS branch first in the
    full-text fallback ladder, (b) uses a reduced entity weight
    (``HYBRID_ENTITY_WEIGHT_V2``), and (c) replaces v1's hard proximity/anchor
    filters with *soft* additive scoring. Each candidate's structured entity
    metadata (coverage, IDF rarity, co-occurrence) is loaded from
    ``chunk_entities`` / ``entities`` and folded into the score along with a
    soft proximity bonus and an anchor penalty (instead of a drop).

    Args:
        state: Shared retrieval state.
        entity_terms: Canonical entity terms for the entity branch.
        fulltext_query: Raw query text for the full-text branch.
        top_k: Number of results to return.
        require_all_entities: If True the entity MATCH joins terms with AND.
        subsets: Optional list of subset names to restrict results to.

    Returns:
        A list of up to ``top_k`` result dicts augmented with debug keys
        (``coverage_score``, ``rarity_score``, ``cooccur_bonus``,
        ``proximity_bonus``, ``keyword_bonus``, ``phrase_bonus``,
        ``hybrid_score``); ``[]`` if nothing matched.
    """

    conn = _get_sqlite_conn(state)

    cur = conn.cursor()

    entity_query = build_entity_match_query(
        entity_terms, require_all=require_all_entities
    )

    raw_ft_query = str(fulltext_query or "").strip()

    phrase_fulltext_query = _build_phrase_query(raw_ft_query)

    strict_fulltext_query = build_fulltext_query(raw_ft_query, require_all=True)

    broad_fulltext_query = build_fulltext_query(raw_ft_query, require_all=False)

    #
    # ----------------------
    # Candidate generation
    # ----------------------
    #

    entity_rows = []

    if entity_query:

        entity_rows = list(
            cur.execute(
                """
              SELECT
                rowid AS lookup_id,
                bm25(
                  entities_fts
                ) AS bm25_score
              FROM entities_fts
              WHERE entities_fts MATCH ?
              ORDER BY bm25_score
              LIMIT ?
              """,
                (entity_query, HYBRID_ENTITY_LIMIT),
            )
        )

    fulltext_rows = []
    fulltext_query_used = ""

    #
    # phrase first
    #
    if phrase_fulltext_query:

        fulltext_rows = list(
            cur.execute(
                """
             SELECT
               rowid AS lookup_id,
               bm25(
                 fulltext_fts
               ) AS bm25_score
             FROM fulltext_fts
             WHERE fulltext_fts MATCH ?
             ORDER BY bm25_score
             LIMIT ?
             """,
                (phrase_fulltext_query, HYBRID_FULLTEXT_LIMIT),
            )
        )

        if fulltext_rows:
            fulltext_query_used = phrase_fulltext_query

    if not fulltext_rows and strict_fulltext_query:

        fulltext_rows = list(
            cur.execute(
                """
              SELECT
               rowid AS lookup_id,
               bm25(
                 fulltext_fts
               ) AS bm25_score
              FROM fulltext_fts
              WHERE fulltext_fts MATCH ?
              ORDER BY bm25_score
              LIMIT ?
              """,
                (strict_fulltext_query, HYBRID_FULLTEXT_LIMIT),
            )
        )

        if fulltext_rows:
            fulltext_query_used = strict_fulltext_query

    if (
        not fulltext_rows
        and broad_fulltext_query
        and broad_fulltext_query != strict_fulltext_query
    ):
        fulltext_rows = list(
            cur.execute(
                """
                SELECT
                  rowid AS lookup_id,
                  bm25(
                    fulltext_fts
                  ) AS bm25_score
                FROM fulltext_fts
                WHERE fulltext_fts MATCH ?
                ORDER BY bm25_score
                LIMIT ?
                """,
                (broad_fulltext_query, HYBRID_FULLTEXT_LIMIT),
            )
        )

        if fulltext_rows:
            fulltext_query_used = broad_fulltext_query

    #
    # ----------------------
    # Merge hybrid
    # ----------------------
    #

    merged = {}

    for rows, field in (
        (entity_rows, "entity_score"),
        (fulltext_rows, "fulltext_score"),
    ):

        for row in rows:

            lookup_id = int(row["lookup_id"])

            score = _fts_positive_score_simple(row["bm25_score"])

            merged.setdefault(
                lookup_id,
                {
                    "entity_score": 0.0,
                    "fulltext_score": 0.0,
                    "raw_score": 0.0,
                },
            )

            merged[lookup_id][field] = max(merged[lookup_id][field], score)

            merged[lookup_id]["raw_score"] = max(
                merged[lookup_id]["raw_score"], score
            )

    ranked = []

    for lookup_id, parts in merged.items():

        hybrid_score = (
            HYBRID_ENTITY_WEIGHT_V2 * parts["entity_score"]
            + HYBRID_FULLTEXT_WEIGHT * parts["fulltext_score"]
        )

        ranked.append(
            (
                lookup_id,
                hybrid_score,
                parts["entity_score"],
                parts["fulltext_score"],
                parts["raw_score"],
            )
        )

    ranked.sort(key=lambda x: x[1], reverse=True)

    prefilter_limit = max(top_k * 20, 200)

    prefilter_limit = min(prefilter_limit, 2000)

    ranked = ranked[:prefilter_limit]

    if not ranked:
        return []

    ids = [r[0] for r in ranked]

    ph = ",".join("?" for _ in ids)

    query = f"""
    SELECT
      lookup_id,
      chunk_id,
      title,
      subset_name,
      domain,
      fulltext_text,
      source_url
    FROM chunks
    WHERE lookup_id IN ({ph})
    """

    params = list(ids)

    if subsets:
        s_ph = ",".join("?" for _ in subsets)

        query += f" AND subset_name IN ({s_ph})"

        params.extend(subsets)

    meta_rows = list(cur.execute(query, params))

    meta_map = {int(r["lookup_id"]): r for r in meta_rows}

    #
    # structured metadata
    #
    entity_df = _load_entity_df(cur, entity_terms)

    candidate_entity_map = _candidate_entities(cur, ids)

    corpus_chunks = _get_cached_chunk_count(state, cur)

    #
    # ----------------------
    # Rescore
    # ----------------------
    #

    rescored = []

    for (
        lookup_id,
        hybrid_score,
        entity_score,
        fulltext_score,
        raw_score,
    ) in ranked:

        row = meta_map.get(lookup_id)

        if row is None:
            continue

        title = str(row["title"] or "")

        fulltext_text = str(row["fulltext_text"] or "")

        if _looks_too_punct_noisy(title, fulltext_text):
            continue

        keyword_bonus = _keyword_overlap_bonus(
            raw_ft_query, title, fulltext_text
        )

        phrase_bonus = _exact_phrase_bonus(raw_ft_query, title, fulltext_text)

        title_penalty = _generic_title_penalty(title)

        #
        # anchor penalty
        #
        anchor_penalty = 0.0

        if entity_terms:
            if not _has_early_anchor(entity_terms, title, fulltext_text):
                anchor_penalty = NO_ANCHOR_PENALTY

        chunk_entities = candidate_entity_map.get(lookup_id, [])

        coverage_score, rarity_score, cooccur_bonus = (
            _entity_structured_score_simple(
                entity_terms, chunk_entities, entity_df, corpus_chunks
            )
        )

        #
        # soft proximity
        #
        proximity_bonus = 0.0

        if len(entity_terms) >= 2:

            if _has_entity_proximity_match(
                entity_terms, fulltext_text, max_chars=250
            ):
                proximity_bonus = PROX_BONUS_NEAR

            elif _has_entity_proximity_match(
                entity_terms, fulltext_text, max_chars=700
            ):
                proximity_bonus = PROX_BONUS_FAR

        adjusted_score = (
            hybrid_score
            + ENTITY_COVERAGE_WEIGHT * coverage_score
            + ENTITY_RARITY_WEIGHT * rarity_score
            + cooccur_bonus
            + proximity_bonus
            - anchor_penalty
            - title_penalty
            + keyword_bonus
            + phrase_bonus
        )

        rescored.append(
            (
                lookup_id,
                adjusted_score,
                entity_score,
                fulltext_score,
                keyword_bonus,
                phrase_bonus,
                raw_score,
                coverage_score,
                rarity_score,
                cooccur_bonus,
                proximity_bonus,
            )
        )

    rescored.sort(key=lambda x: x[1], reverse=True)

    rescored = rescored[:top_k]

    results = []

    for (
        lookup_id,
        adjusted_score,
        entity_score,
        fulltext_score,
        keyword_bonus,
        phrase_bonus,
        raw_score,
        coverage_score,
        rarity_score,
        cooccur_bonus,
        proximity_bonus,
    ) in rescored:

        row = meta_map[lookup_id]

        result = _sqlite_row_to_result(
            row, adjusted_score, entity_score, fulltext_score, raw_score
        )

        #
        # Debug
        #
        result["coverage_score"] = float(coverage_score)

        result["rarity_score"] = float(rarity_score)

        result["cooccur_bonus"] = float(cooccur_bonus)

        result["proximity_bonus"] = float(proximity_bonus)

        result["keyword_bonus"] = float(keyword_bonus)

        result["phrase_bonus"] = float(phrase_bonus)

        result["hybrid_score"] = float(adjusted_score)

        results.append(result)

    return results


# ============================================================
# V3 tunables
# ============================================================

HYBRID_ENTITY_WEIGHT_V3 = HYBRID_ENTITY_WEIGHT * 0.75

ENTITY_COVERAGE_WEIGHT = 0.55
ENTITY_RARITY_WEIGHT = 0.30

COOCCUR_BONUS = 0.15

PROX_BONUS_NEAR = 0.15
PROX_BONUS_FAR = 0.05

NO_ANCHOR_PENALTY = 0.25
IDF_CAP = 3.0

PREDICATE_WEIGHT = 0.50
CONTENT_OVERLAP_WEIGHT = 0.25
ENTITY_PRED_MISMATCH_PENALTY = 0.60

RELATION_MODE_MIN_PRED_SUPPORT = 0.15


ENTITY_TYPE_WEIGHT = {
    "person": 1.0,
    "event": 1.0,
    "organization": 0.7,
    "work": 0.5,
    "location": 0.4,
    "date": 0.2,
}


# ============================================================
# Broad relation/predicate vocabulary
# ============================================================

RELATION_LEMMA_EXPANSIONS: Dict[str, Set[str]] = {
    # violence / death / harm
    "kill": {
        "kill",
        "killed",
        "killing",
        "murder",
        "murdered",
        "assassinate",
        "assassinated",
        "slay",
        "slain",
        "execute",
        "executed",
        "shoot",
        "shot",
    },
    "murder": {
        "murder",
        "murdered",
        "kill",
        "killed",
        "assassinate",
        "assassinated",
        "shot",
    },
    "assassinate": {
        "assassinate",
        "assassinated",
        "murder",
        "murdered",
        "kill",
        "killed",
        "shot",
    },
    "shoot": {
        "shoot",
        "shot",
        "gunman",
        "gunmen",
        "fire",
        "fired",
        "wound",
        "wounded",
    },
    "poison": {"poison", "poisoned", "toxin", "toxic", "overdose"},
    "attack": {
        "attack",
        "attacked",
        "assault",
        "assaulted",
        "strike",
        "struck",
    },
    "harm": {"harm", "harmed", "injure", "injured", "damage", "damaged"},
    "die": {"die", "died", "death", "dead", "killed", "fatal"},
    # agency / responsibility / causation
    "cause": {
        "cause",
        "caused",
        "causing",
        "lead",
        "led",
        "trigger",
        "triggered",
        "produce",
        "produced",
        "result",
        "resulted",
    },
    "trigger": {"trigger", "triggered", "cause", "caused", "spark", "sparked"},
    "lead": {"lead", "led", "cause", "caused", "resulted"},
    "blame": {"blame", "blamed", "responsible", "culpable", "fault"},
    "responsible": {
        "responsible",
        "culpable",
        "accountable",
        "blame",
        "blamed",
    },
    # funding / money / support
    "fund": {
        "fund",
        "funded",
        "funding",
        "finance",
        "financed",
        "financing",
        "bankroll",
        "bankrolled",
        "sponsor",
        "sponsored",
    },
    "finance": {
        "finance",
        "financed",
        "fund",
        "funded",
        "bankroll",
        "bankrolled",
    },
    "pay": {"pay", "paid", "payment", "payments", "finance", "funded"},
    "sponsor": {"sponsor", "sponsored", "funded", "backed", "supported"},
    "back": {"back", "backed", "support", "supported", "sponsor", "sponsored"},
    # command / authorization / planning
    "order": {
        "order",
        "ordered",
        "command",
        "commanded",
        "authorize",
        "authorized",
        "direct",
        "directed",
        "instruct",
        "instructed",
    },
    "authorize": {
        "authorize",
        "authorized",
        "approve",
        "approved",
        "sanction",
        "sanctioned",
    },
    "direct": {
        "direct",
        "directed",
        "order",
        "ordered",
        "command",
        "commanded",
    },
    "plan": {
        "plan",
        "planned",
        "plot",
        "plotted",
        "scheme",
        "schemed",
        "conspire",
        "conspired",
    },
    "organize": {
        "organize",
        "organized",
        "coordinate",
        "coordinated",
        "orchestrate",
        "orchestrated",
    },
    "orchestrate": {
        "orchestrate",
        "orchestrated",
        "coordinate",
        "coordinated",
        "organize",
        "organized",
    },
    # control / influence / manipulation
    "control": {
        "control",
        "controlled",
        "influence",
        "influenced",
        "manipulate",
        "manipulated",
        "steer",
        "steered",
    },
    "influence": {
        "influence",
        "influenced",
        "pressure",
        "pressured",
        "lobby",
        "lobbied",
        "shape",
        "shaped",
    },
    "manipulate": {
        "manipulate",
        "manipulated",
        "rig",
        "rigged",
        "distort",
        "distorted",
    },
    "coerce": {"coerce", "coerced", "force", "forced", "pressure", "pressured"},
    # creation / origin / invention / founding
    "create": {
        "create",
        "created",
        "found",
        "founded",
        "establish",
        "established",
        "form",
        "formed",
        "invent",
        "invented",
    },
    "found": {
        "found",
        "founded",
        "create",
        "created",
        "establish",
        "established",
    },
    "establish": {
        "establish",
        "established",
        "create",
        "created",
        "found",
        "founded",
    },
    "invent": {
        "invent",
        "invented",
        "develop",
        "developed",
        "create",
        "created",
    },
    "develop": {"develop", "developed", "build", "built", "create", "created"},
    # discovery / revelation / investigation
    "discover": {
        "discover",
        "discovered",
        "find",
        "found",
        "uncover",
        "uncovered",
        "reveal",
        "revealed",
    },
    "reveal": {
        "reveal",
        "revealed",
        "disclose",
        "disclosed",
        "expose",
        "exposed",
        "uncover",
        "uncovered",
    },
    "investigate": {
        "investigate",
        "investigated",
        "probe",
        "probed",
        "inquiry",
        "inquiries",
    },
    "expose": {
        "expose",
        "exposed",
        "reveal",
        "revealed",
        "uncover",
        "uncovered",
    },
    # accusation / claim / testimony
    "accuse": {
        "accuse",
        "accused",
        "allege",
        "alleged",
        "claim",
        "claimed",
        "charge",
        "charged",
    },
    "allege": {"allege", "alleged", "accuse", "accused", "claim", "claimed"},
    "claim": {"claim", "claimed", "allege", "alleged", "assert", "asserted"},
    "testify": {"testify", "testified", "testimony", "witness", "witnessed"},
    "admit": {
        "admit",
        "admitted",
        "confess",
        "confessed",
        "acknowledge",
        "acknowledged",
    },
    # concealment / coverup / suppression
    "hide": {
        "hide",
        "hid",
        "hidden",
        "conceal",
        "concealed",
        "cover",
        "covered",
        "suppress",
        "suppressed",
    },
    "conceal": {
        "conceal",
        "concealed",
        "hide",
        "hidden",
        "coverup",
        "cover-up",
    },
    "suppress": {
        "suppress",
        "suppressed",
        "censor",
        "censored",
        "bury",
        "buried",
    },
    "censor": {"censor", "censored", "suppress", "suppressed", "ban", "banned"},
    "cover": {
        "cover",
        "covered",
        "coverup",
        "cover-up",
        "conceal",
        "concealed",
    },
    # relationship / connection
    "connect": {
        "connect",
        "connected",
        "link",
        "linked",
        "associate",
        "associated",
        "relate",
        "related",
    },
    "link": {
        "link",
        "linked",
        "connect",
        "connected",
        "tie",
        "tied",
        "associate",
        "associated",
    },
    "associate": {
        "associate",
        "associated",
        "link",
        "linked",
        "connect",
        "connected",
    },
    "meet": {"meet", "met", "meeting", "encounter", "encountered"},
    "work": {
        "work",
        "worked",
        "collaborate",
        "collaborated",
        "cooperate",
        "cooperated",
    },
    # membership / affiliation
    "join": {"join", "joined", "member", "membership", "belong", "belonged"},
    "belong": {"belong", "belonged", "member", "membership", "affiliated"},
    "affiliate": {
        "affiliate",
        "affiliated",
        "associate",
        "associated",
        "member",
    },
    # legal / official actions
    "arrest": {"arrest", "arrested", "detain", "detained", "custody"},
    "charge": {"charge", "charged", "indict", "indicted", "accuse", "accused"},
    "convict": {"convict", "convicted", "sentence", "sentenced", "guilty"},
    "sue": {"sue", "sued", "lawsuit", "litigation"},
    "ban": {"ban", "banned", "prohibit", "prohibited", "outlaw", "outlawed"},
    # transmission / spread / dissemination
    "spread": {
        "spread",
        "spreading",
        "transmit",
        "transmitted",
        "disseminate",
        "disseminated",
        "circulate",
        "circulated",
    },
    "publish": {
        "publish",
        "published",
        "release",
        "released",
        "distribute",
        "distributed",
    },
    "leak": {"leak", "leaked", "disclose", "disclosed", "release", "released"},
    # comparison / identity / attribution
    "be": {"is", "was", "were", "are", "be", "being", "become", "became"},
    "identify": {
        "identify",
        "identified",
        "name",
        "named",
        "recognize",
        "recognized",
    },
    "name": {"name", "named", "identify", "identified"},
}


RELATION_VERBS = set(RELATION_LEMMA_EXPANSIONS.keys())


# ============================================================
# Helpers
# ============================================================


def _build_phrase_query(raw_query: str) -> str:
    """Build a quoted FTS5 phrase query from a raw query.

    Args:
        raw_query: The raw query string.

    Returns:
        The query wrapped in double quotes (inner quotes escaped) as an FTS5
        phrase, or an empty string if the query has fewer than 4 words.
    """
    q = str(raw_query or "").strip()
    if len(q.split()) < 4:
        return ""
    # Escape quotes for FTS phrase use.
    q = q.replace('"', '""')
    return f'"{q}"'


def _safe_lower_text(text: str) -> str:
    """Coerce a value to a lowercased string, treating ``None`` as empty.

    Args:
        text: Any value, possibly ``None``.

    Returns:
        The lowercased string form (empty string for ``None``).
    """
    return str(text or "").lower()


def _contains_any_term(text_lc: str, terms: Set[str]) -> bool:
    """Check whether any of ``terms`` occurs as a word in lowercased text.

    Args:
        text_lc: Already-lowercased text to search.
        terms: Candidate terms to look for.

    Returns:
        True if any non-empty term matches on a word boundary.
    """
    for term in terms:
        if not term:
            continue
        # Conservative word-ish boundary for normal words; substring fallback for hyphenated forms.
        if re.search(r"\b" + re.escape(term.lower()) + r"\b", text_lc):
            return True
    return False


def _get_cached_chunk_count(state: RetrievalState, cur: sqlite3.Cursor) -> int:
    """Return the total ``chunks`` row count, caching it on ``state``.

    Args:
        state: Shared retrieval state; the count is memoized on
            ``state._cached_chunk_count``.
        cur: An open SQLite cursor.

    Returns:
        The total number of rows in the ``chunks`` table.
    """
    cached = getattr(state, "_cached_chunk_count", None)
    if cached:
        return int(cached)

    cur.execute("SELECT COUNT(*) AS n FROM chunks")
    n = int(cur.fetchone()["n"])
    state._cached_chunk_count = n
    return n


def _load_entity_df(
    cur: sqlite3.Cursor, query_entities: List[str]
) -> Dict[str, int]:
    """Load per-entity document frequencies for IDF rarity scoring.

    Args:
        cur: An open SQLite cursor.
        query_entities: Canonical entity names to look up.

    Returns:
        A dict mapping each found ``canonical_name`` to its document
        frequency (distinct chunk count); empty if no query entities.
    """
    if not query_entities:
        return {}

    ph = ",".join("?" * len(query_entities))

    rows = list(
        cur.execute(
            f"""
            SELECT
                e.canonical_name,
                COUNT(DISTINCT ce.chunk_lookup_id) AS df
            FROM entities e
            JOIN chunk_entities ce
              ON e.entity_id = ce.entity_id
            WHERE e.canonical_name IN ({ph})
            GROUP BY e.entity_id
            """,
            query_entities,
        )
    )

    return {r["canonical_name"]: int(r["df"]) for r in rows}


def _candidate_entities(
    cur: sqlite3.Cursor, ids: List[int]
) -> Dict[int, List[Tuple[str, str]]]:
    """Fetch the entities attached to each candidate chunk.

    Args:
        cur: An open SQLite cursor.
        ids: Chunk ``lookup_id`` values to fetch entities for.

    Returns:
        A dict mapping each chunk ``lookup_id`` to a list of
        ``(canonical_name, type)`` entity pairs; empty if ``ids`` is empty.
    """
    if not ids:
        return {}

    ph = ",".join("?" * len(ids))

    rows = list(
        cur.execute(
            f"""
            SELECT
                ce.chunk_lookup_id,
                e.canonical_name,
                e.type
            FROM chunk_entities ce
            JOIN entities e
              ON ce.entity_id = e.entity_id
            WHERE ce.chunk_lookup_id IN ({ph})
            """,
            ids,
        )
    )

    out: Dict[int, List[Tuple[str, str]]] = {}

    for r in rows:
        cid = int(r["chunk_lookup_id"])
        out.setdefault(cid, []).append((r["canonical_name"], r["type"]))

    return out


def _entity_structured_score_simple(
    query_entities: List[str],
    chunk_entities: List[Tuple[str, str]],
    entity_df: Dict[str, int],
    corpus_chunks: int,
) -> Tuple[float, float, float]:
    """Score a chunk's entity match (legacy 3-tuple version).

    LEGACY / SIMPLE VERSION (original 3-return). Used in early v2; lacks the
    matched-entity count of ``_entity_structured_score``.

    Args:
        query_entities: Canonical entity terms from the query.
        chunk_entities: ``(canonical_name, type)`` pairs attached to the chunk.
        entity_df: Per-entity document frequencies for IDF computation.
        corpus_chunks: Total chunk count in the corpus.

    Returns:
        A ``(coverage_score, rarity_score, cooccur_bonus)`` tuple, all zeros
        if there are no query entities or no matched entities. ``coverage_score``
        is ``coverage ** 1.5``; ``rarity_score`` is the type-weighted mean IDF
        of matched entities; ``cooccur_bonus`` is ``COOCCUR_BONUS`` when 2+
        entities matched, else 0.0.
    """
    if not query_entities:
        return 0.0, 0.0, 0.0

    chunk_names = {e[0] for e in chunk_entities}
    matched = [e for e in query_entities if e in chunk_names]

    if not matched:
        return 0.0, 0.0, 0.0

    coverage = len(matched) / len(query_entities)
    coverage_score = coverage**1.5

    rarity = 0.0
    type_map = dict(chunk_entities)
    safe_corpus_chunks = max(int(corpus_chunks or 1), 1)

    for e in matched:
        df = max(entity_df.get(e, 1), 1)
        idf = (
            math.log(safe_corpus_chunks / df)
            if safe_corpus_chunks > df
            else 0.0
        )
        idf = min(idf, IDF_CAP)
        rarity += ENTITY_TYPE_WEIGHT.get(type_map.get(e), 0.5) * idf

    rarity /= max(len(matched), 1)
    cooccur_bonus = COOCCUR_BONUS if len(matched) >= 2 else 0.0

    return coverage_score, rarity, cooccur_bonus


def _entity_structured_score(
    query_entities: List[str],
    chunk_entities: List[Tuple[str, str]],
    entity_df: Dict[str, int],
    corpus_chunks: int,
) -> Tuple[float, float, float, int]:
    """Score a chunk's entity match (enhanced 4-tuple version).

    ENHANCED VERSION (current 4-return). Identical scoring to
    ``_entity_structured_score_simple`` but additionally returns the matched
    entity count, used for null-evidence detection and debugging.

    Args:
        query_entities: Canonical entity terms from the query.
        chunk_entities: ``(canonical_name, type)`` pairs attached to the chunk.
        entity_df: Per-entity document frequencies for IDF computation.
        corpus_chunks: Total chunk count in the corpus.

    Returns:
        A ``(coverage_score, rarity_score, cooccur_bonus, matched_count)``
        tuple, ``(0.0, 0.0, 0.0, 0)`` if there are no query entities or no
        matched entities.
    """
    if not query_entities:
        return 0.0, 0.0, 0.0, 0

    chunk_names = {e[0] for e in chunk_entities}
    matched = [e for e in query_entities if e in chunk_names]

    if not matched:
        return 0.0, 0.0, 0.0, 0

    coverage = len(matched) / len(query_entities)
    coverage_score = coverage**1.5

    rarity = 0.0
    type_map = dict(chunk_entities)
    safe_corpus_chunks = max(int(corpus_chunks or 1), 1)

    for e in matched:
        df = max(entity_df.get(e, 1), 1)
        idf = (
            math.log(safe_corpus_chunks / df)
            if safe_corpus_chunks > df
            else 0.0
        )
        idf = min(idf, IDF_CAP)
        rarity += ENTITY_TYPE_WEIGHT.get(type_map.get(e), 0.5) * idf

    rarity /= max(len(matched), 1)
    cooccur_bonus = COOCCUR_BONUS if len(matched) >= 2 else 0.0

    return coverage_score, rarity, cooccur_bonus, len(matched)


def _analyze_query_spacy(nlp, raw_query: str) -> Dict[str, Any]:
    """Analyze a query's structure with spaCy into a soft profile.

    Finds the root verb (skipping copulas/auxiliaries), expands it via
    ``RELATION_LEMMA_EXPANSIONS``, extracts content lemmas, and infers
    whether the query is relation-seeking.

    Args:
        nlp: A loaded spaCy ``Language`` pipeline.
        raw_query: The raw query string.

    Returns:
        A dict with keys: ``predicate`` (str or None), ``predicate_variants``
        (Set[str]), ``content_terms`` (Set[str]), ``relation_mode`` (bool),
        and ``query_type`` (``"empty"`` / ``"relation_factoid"`` /
        ``"exploratory"``).
    """
    q = str(raw_query or "").strip()

    if not q:
        return {
            "predicate": None,
            "predicate_variants": set(),
            "content_terms": set(),
            "relation_mode": False,
            "query_type": "empty",
        }

    doc = nlp(q)

    root = None
    for tok in doc:
        if tok.dep_ == "ROOT":
            root = tok
            break

    predicate = root.lemma_.lower() if root is not None else None

    # If root is a copula/helper, try to find a meaningful verb.
    if predicate in {"be", "do", "have"}:
        for tok in doc:
            if tok.pos_ == "VERB" and tok.lemma_.lower() not in {
                "be",
                "do",
                "have",
            }:
                predicate = tok.lemma_.lower()
                break

    wh_terms = {
        tok.lower_
        for tok in doc
        if tok.tag_ in {"WP", "WDT", "WRB"}
        or tok.lower_ in {"who", "what", "when", "where", "why", "how", "which"}
    }

    content_terms: Set[str] = set()
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.like_num:
            continue

        if tok.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"}:
            lemma = tok.lemma_.lower().strip()
            if lemma and len(lemma) > 1:
                content_terms.add(lemma)

    # Remove predicate variants from content terms later? Keep predicate in content terms for general overlap.
    predicate_variants: Set[str] = set()

    if predicate:
        predicate_variants = RELATION_LEMMA_EXPANSIONS.get(
            predicate, {predicate}
        )

    relation_mode = False

    # Explicit relation predicate.
    if predicate in RELATION_VERBS:
        relation_mode = True

    # WH factoid with a meaningful verb is often relation-seeking.
    if wh_terms and predicate and predicate not in {"be", "do", "have"}:
        relation_mode = True

    # Query phrases like "who killed", "what caused", etc.
    q_lc = q.lower()
    if re.search(r"\b(who|what|which|when|where|why|how)\b", q_lc):
        if predicate and predicate not in {"be", "do", "have"}:
            relation_mode = True

    query_type = "relation_factoid" if relation_mode else "exploratory"

    return {
        "predicate": predicate,
        "predicate_variants": predicate_variants,
        "content_terms": content_terms,
        "relation_mode": relation_mode,
        "query_type": query_type,
    }


def _predicate_support_score(
    predicate_variants: Set[str],
    title: str,
    fulltext_text: str,
) -> float:
    """Score how well a chunk supports the query's predicate/relation.

    Args:
        predicate_variants: Lemma variants of the query predicate.
        title: The chunk title.
        fulltext_text: The chunk body text.

    Returns:
        A score in ``[0.0, 1.0]`` equal to ``min(hits / 2.0, 1.0)`` where
        ``hits`` is the count of distinct predicate variants found.
    """
    if not predicate_variants:
        return 0.0

    text_lc = _safe_lower_text(f"{title} {fulltext_text}")

    hits = 0
    for term in predicate_variants:
        if not term:
            continue
        if re.search(r"\b" + re.escape(term.lower()) + r"\b", text_lc):
            hits += 1

    return min(hits / 2.0, 1.0)


def _content_overlap_score(
    content_terms: Set[str],
    title: str,
    fulltext_text: str,
) -> float:
    """Score the fraction of query content terms present in a chunk.

    Args:
        content_terms: Content lemmas extracted from the query.
        title: The chunk title.
        fulltext_text: The chunk body text.

    Returns:
        The fraction of ``content_terms`` matched on a word boundary, in
        ``[0.0, 1.0]``; ``0.0`` when there are no content terms.
    """
    if not content_terms:
        return 0.0

    text_lc = _safe_lower_text(f"{title} {fulltext_text}")

    hits = 0
    for term in content_terms:
        if re.search(r"\b" + re.escape(term.lower()) + r"\b", text_lc):
            hits += 1

    return hits / max(len(content_terms), 1)


# ============================================================
# Hybrid search v3
# ============================================================


def _hybrid_search_sqlite3(
    state: RetrievalState,
    *,
    entity_terms: List[str],
    fulltext_query: str,
    top_k: int,
    require_all_entities: bool = True,
    subsets: List[str] = None,
) -> List[Dict[str, Any]]:
    """Run the v3 hybrid retrieval algorithm (predicate-aware rescoring).

    Extends v2 with spaCy query-structure analysis (``_analyze_query_spacy``):
    it adds a predicate-support term and a content-overlap term to the score,
    and applies a "Gandhi-style" mismatch penalty when a chunk matches the
    query entities strongly but carries no predicate evidence in relation
    mode. Also computes top-N null-evidence diagnostics for relation queries.

    Args:
        state: Shared retrieval state.
        entity_terms: Canonical entity terms for the entity branch.
        fulltext_query: Raw query text for the full-text branch.
        top_k: Number of results to return.
        require_all_entities: If True the entity MATCH joins terms with AND.
        subsets: Optional list of subset names to restrict results to.

    Returns:
        A list of up to ``top_k`` result dicts augmented with v3 debug keys
        (``predicate_support``, ``content_overlap``, ``mismatch_penalty``,
        ``anchor_penalty``, ``matched_entity_count``, ``query_predicate``,
        ``query_relation_mode``, ``query_type``, plus ``null_evidence_flag``
        and mean predicate/coverage diagnostics for relation queries);
        ``[]`` if nothing matched.
    """

    conn = _get_sqlite_conn(state)
    cur = conn.cursor()

    entity_terms = entity_terms or []

    entity_query = build_entity_match_query(
        entity_terms,
        require_all=require_all_entities,
    )

    raw_ft_query = str(fulltext_query or "").strip()

    query_struct = _analyze_query_spacy(state.nlp, raw_ft_query)
    predicate = query_struct["predicate"]
    predicate_variants = query_struct["predicate_variants"]
    relation_mode = bool(query_struct["relation_mode"])
    content_terms = query_struct["content_terms"]
    query_type = query_struct["query_type"]

    phrase_fulltext_query = _build_phrase_query(raw_ft_query)

    strict_fulltext_query = build_fulltext_query(
        raw_ft_query,
        require_all=True,
    )

    broad_fulltext_query = build_fulltext_query(
        raw_ft_query,
        require_all=False,
    )

    rag_logger.info(
        "Hybrid3 query. raw=%s entity_terms=%s predicate=%s relation_mode=%s query_type=%s content_terms=%s",
        raw_ft_query,
        entity_terms,
        predicate,
        relation_mode,
        query_type,
        sorted(list(content_terms))[:20],
    )

    # ----------------------------
    # 1) Entity search
    # ----------------------------
    entity_rows = []

    if entity_query:
        entity_rows = list(
            cur.execute(
                """
                SELECT rowid AS lookup_id, bm25(entities_fts) AS bm25_score
                FROM entities_fts
                WHERE entities_fts MATCH ?
                ORDER BY bm25_score
                LIMIT ?
                """,
                (entity_query, HYBRID_ENTITY_LIMIT),
            )
        )

    # ----------------------------
    # 2) Fulltext search
    # phrase -> strict AND -> broad OR
    # ----------------------------
    fulltext_rows = []
    fulltext_query_used = ""

    if phrase_fulltext_query:
        fulltext_rows = list(
            cur.execute(
                """
                SELECT rowid AS lookup_id, bm25(fulltext_fts) AS bm25_score
                FROM fulltext_fts
                WHERE fulltext_fts MATCH ?
                ORDER BY bm25_score
                LIMIT ?
                """,
                (phrase_fulltext_query, HYBRID_FULLTEXT_LIMIT),
            )
        )

        if fulltext_rows:
            fulltext_query_used = phrase_fulltext_query

    if not fulltext_rows and strict_fulltext_query:
        fulltext_rows = list(
            cur.execute(
                """
                SELECT rowid AS lookup_id, bm25(fulltext_fts) AS bm25_score
                FROM fulltext_fts
                WHERE fulltext_fts MATCH ?
                ORDER BY bm25_score
                LIMIT ?
                """,
                (strict_fulltext_query, HYBRID_FULLTEXT_LIMIT),
            )
        )

        if fulltext_rows:
            fulltext_query_used = strict_fulltext_query

    if (
        not fulltext_rows
        and broad_fulltext_query
        and broad_fulltext_query != strict_fulltext_query
    ):
        fulltext_rows = list(
            cur.execute(
                """
                SELECT rowid AS lookup_id, bm25(fulltext_fts) AS bm25_score
                FROM fulltext_fts
                WHERE fulltext_fts MATCH ?
                ORDER BY bm25_score
                LIMIT ?
                """,
                (broad_fulltext_query, HYBRID_FULLTEXT_LIMIT),
            )
        )

        if fulltext_rows:
            fulltext_query_used = broad_fulltext_query

    # ----------------------------
    # 3) Merge branch scores
    # ----------------------------
    merged: Dict[int, Dict[str, float]] = {}

    for row in entity_rows:
        lookup_id = int(row["lookup_id"])
        score = _fts_positive_score_simple(row["bm25_score"])

        merged.setdefault(
            lookup_id,
            {
                "entity_score": 0.0,
                "fulltext_score": 0.0,
                "raw_score": 0.0,
            },
        )

        merged[lookup_id]["entity_score"] = max(
            merged[lookup_id]["entity_score"],
            score,
        )

        merged[lookup_id]["raw_score"] = max(
            merged[lookup_id]["raw_score"],
            score,
        )

    for row in fulltext_rows:
        lookup_id = int(row["lookup_id"])
        score = _fts_positive_score_simple(row["bm25_score"])

        merged.setdefault(
            lookup_id,
            {
                "entity_score": 0.0,
                "fulltext_score": 0.0,
                "raw_score": 0.0,
            },
        )

        merged[lookup_id]["fulltext_score"] = max(
            merged[lookup_id]["fulltext_score"],
            score,
        )

        merged[lookup_id]["raw_score"] = max(
            merged[lookup_id]["raw_score"],
            score,
        )

    # ----------------------------
    # 4) Initial ranking
    # ----------------------------
    ranked = []

    for lookup_id, parts in merged.items():
        hybrid_score = (
            HYBRID_ENTITY_WEIGHT_V3 * parts["entity_score"]
            + HYBRID_FULLTEXT_WEIGHT * parts["fulltext_score"]
        )

        ranked.append(
            (
                lookup_id,
                hybrid_score,
                parts["entity_score"],
                parts["fulltext_score"],
                parts["raw_score"],
            )
        )

    ranked.sort(key=lambda x: x[1], reverse=True)

    prefilter_limit = max(top_k * 20, 200)
    prefilter_limit = min(prefilter_limit, 2000)
    ranked = ranked[:prefilter_limit]

    if not ranked:
        rag_logger.info(
            "Hybrid3 empty. entity_terms=%s entity_query=%s fulltext_query_used=%s entity_hits=%d fulltext_hits=%d",
            entity_terms,
            entity_query,
            fulltext_query_used,
            len(entity_rows),
            len(fulltext_rows),
        )
        return []

    ids = [r[0] for r in ranked]

    # ----------------------------
    # 5) Pull chunk metadata
    # ----------------------------
    placeholders = ",".join("?" for _ in ids)

    query = f"""
        SELECT lookup_id, chunk_id, title, subset_name, domain, fulltext_text, source_url
        FROM chunks
        WHERE lookup_id IN ({placeholders})
    """

    params = list(ids)

    if subsets:
        subset_placeholders = ",".join("?" for _ in subsets)
        query += f" AND subset_name IN ({subset_placeholders})"
        params.extend(subsets)

        rag_logger.info("Hybrid3 subsets used: %s", subsets)
    else:
        rag_logger.info("Hybrid3 no subsets used")

    meta_rows = list(cur.execute(query, params))
    meta_map = {int(r["lookup_id"]): r for r in meta_rows}

    # ----------------------------
    # 6) Structured metadata
    # ----------------------------
    entity_df = _load_entity_df(cur, entity_terms)
    candidate_entity_map = _candidate_entities(cur, ids)
    corpus_chunks = _get_cached_chunk_count(state, cur)

    # ----------------------------
    # 7) Post-filter + v3 rescoring
    # ----------------------------
    rescored = []
    dropped_noisy = 0

    for (
        lookup_id,
        hybrid_score,
        entity_score,
        fulltext_score,
        raw_score,
    ) in ranked:
        row = meta_map.get(lookup_id)

        if row is None:
            continue

        title = str(row["title"] or "")
        fulltext_text = str(row["fulltext_text"] or "")

        if _looks_too_punct_noisy(title, fulltext_text):
            dropped_noisy += 1
            continue

        keyword_bonus = _keyword_overlap_bonus(
            raw_ft_query,
            title,
            fulltext_text,
        )

        phrase_bonus = _exact_phrase_bonus(
            raw_ft_query,
            title,
            fulltext_text,
        )

        title_penalty = _generic_title_penalty(title)

        anchor_penalty = 0.0

        if entity_terms:
            if not _has_early_anchor(entity_terms, title, fulltext_text):
                anchor_penalty = NO_ANCHOR_PENALTY

        chunk_entities = candidate_entity_map.get(lookup_id, [])

        (
            coverage_score,
            rarity_score,
            cooccur_bonus,
            matched_entity_count,
        ) = _entity_structured_score(
            entity_terms,
            chunk_entities,
            entity_df,
            corpus_chunks,
        )

        proximity_bonus = 0.0

        if len(entity_terms) >= 2:
            if _has_entity_proximity_match(
                entity_terms,
                fulltext_text,
                max_chars=250,
            ):
                proximity_bonus = PROX_BONUS_NEAR

            elif _has_entity_proximity_match(
                entity_terms,
                fulltext_text,
                max_chars=700,
            ):
                proximity_bonus = PROX_BONUS_FAR

        predicate_support = _predicate_support_score(
            predicate_variants,
            title,
            fulltext_text,
        )

        content_overlap = _content_overlap_score(
            content_terms,
            title,
            fulltext_text,
        )

        mismatch_penalty = 0.0

        # Gandhi-style detector:
        # entity match is strong, but relation/predicate evidence is absent.
        if relation_mode and entity_terms:
            if (
                coverage_score >= 0.80
                and predicate_support < RELATION_MODE_MIN_PRED_SUPPORT
            ):
                mismatch_penalty = ENTITY_PRED_MISMATCH_PENALTY

        adjusted_score = (
            hybrid_score
            + ENTITY_COVERAGE_WEIGHT * coverage_score
            + ENTITY_RARITY_WEIGHT * rarity_score
            + cooccur_bonus
            + proximity_bonus
            + PREDICATE_WEIGHT * predicate_support
            + CONTENT_OVERLAP_WEIGHT * content_overlap
            + keyword_bonus
            + phrase_bonus
            - mismatch_penalty
            - anchor_penalty
            - title_penalty
        )

        rescored.append(
            (
                lookup_id,
                adjusted_score,
                entity_score,
                fulltext_score,
                keyword_bonus,
                phrase_bonus,
                raw_score,
                coverage_score,
                rarity_score,
                cooccur_bonus,
                proximity_bonus,
                predicate_support,
                content_overlap,
                mismatch_penalty,
                anchor_penalty,
                matched_entity_count,
            )
        )

    rescored.sort(key=lambda x: x[1], reverse=True)
    rescored = rescored[:top_k]

    # ----------------------------
    # 8) Final results
    # ----------------------------
    results = []

    for (
        lookup_id,
        adjusted_score,
        entity_score,
        fulltext_score,
        keyword_bonus,
        phrase_bonus,
        raw_score,
        coverage_score,
        rarity_score,
        cooccur_bonus,
        proximity_bonus,
        predicate_support,
        content_overlap,
        mismatch_penalty,
        anchor_penalty,
        matched_entity_count,
    ) in rescored:

        row = meta_map.get(lookup_id)

        if row is None:
            continue

        result = _sqlite_row_to_result(
            row,
            adjusted_score,
            entity_score,
            fulltext_score,
            raw_score,
        )

        result["hybrid_score"] = float(adjusted_score)
        result["search_closeness"] = float(adjusted_score)

        result["entity_score"] = float(entity_score)
        result["fulltext_score"] = float(fulltext_score)
        result["raw_score"] = float(raw_score)

        result["keyword_bonus"] = float(keyword_bonus)
        result["phrase_bonus"] = float(phrase_bonus)

        result["coverage_score"] = float(coverage_score)
        result["rarity_score"] = float(rarity_score)
        result["cooccur_bonus"] = float(cooccur_bonus)
        result["proximity_bonus"] = float(proximity_bonus)

        result["predicate_support"] = float(predicate_support)
        result["content_overlap"] = float(content_overlap)
        result["mismatch_penalty"] = float(mismatch_penalty)
        result["anchor_penalty"] = float(anchor_penalty)
        result["matched_entity_count"] = int(matched_entity_count)

        result["query_predicate"] = predicate or ""
        result["query_relation_mode"] = bool(relation_mode)
        result["query_type"] = query_type

        result["entity_query_used"] = entity_query or ""
        result["fulltext_query_used"] = fulltext_query_used or ""
        result["subset"] = row["subset_name"] or ""

        results.append(result)

    # ----------------------------
    # 9) Null-evidence diagnostics
    # ----------------------------
    null_evidence_flag = False

    if relation_mode and results:
        top_n = results[: min(5, len(results))]

        mean_predicate_support = sum(
            r.get("predicate_support", 0.0) for r in top_n
        ) / max(len(top_n), 1)

        mean_coverage = sum(r.get("coverage_score", 0.0) for r in top_n) / max(
            len(top_n), 1
        )

        if (
            mean_coverage >= 0.70
            and mean_predicate_support < RELATION_MODE_MIN_PRED_SUPPORT
        ):
            null_evidence_flag = True

        for r in results:
            r["null_evidence_flag"] = bool(null_evidence_flag)
            r["mean_top_predicate_support"] = float(mean_predicate_support)
            r["mean_top_coverage"] = float(mean_coverage)

    rag_logger.info(
        "Hybrid3 final. entity_terms=%s predicate=%s relation_mode=%s entity_query=%s fulltext_query_used=%s "
        "entity_hits=%d fulltext_hits=%d kept=%d dropped_noisy=%d null_evidence_flag=%s",
        entity_terms,
        predicate,
        relation_mode,
        entity_query,
        fulltext_query_used,
        len(entity_rows),
        len(fulltext_rows),
        len(results),
        dropped_noisy,
        null_evidence_flag,
    )

    return results


# ------------------ V4: MIN-GATE WRAPPER + HELPERS ------------------


def _detect_fts_branch(state: RetrievalState, fulltext_query: str) -> str:
    """
    Determine which FTS branch (phrase / strict_and / broad_or / none) would
    have produced the first non-empty match for `fulltext_query`. Mirrors the
    fallback ladder inside _hybrid_search_sqlite{,2,3} but only runs LIMIT 1
    probes — total cost is a few ms per call.

    Used by the v4 min-gate's second condition: queries with no canonical
    entities AND only broad-OR matches are very likely off-corpus probes.

    Args:
        state: Shared retrieval state.
        fulltext_query: The raw full-text query.

    Returns:
        One of ``"phrase"``, ``"strict_and"``, ``"broad_or"`` or ``"none"`` —
        the first branch (in fallback order) that produced a match.
    """
    raw = str(fulltext_query or "").strip()
    if not raw:
        return "none"

    conn = _get_sqlite_conn(state)
    cur = conn.cursor()

    phrase_q = _build_phrase_query(raw)
    if phrase_q:
        row = cur.execute(
            "SELECT 1 FROM fulltext_fts WHERE fulltext_fts MATCH ? LIMIT 1",
            (phrase_q,),
        ).fetchone()
        if row:
            return "phrase"

    strict_q = build_fulltext_query(raw, require_all=True)
    if strict_q:
        row = cur.execute(
            "SELECT 1 FROM fulltext_fts WHERE fulltext_fts MATCH ? LIMIT 1",
            (strict_q,),
        ).fetchone()
        if row:
            return "strict_and"

    broad_q = build_fulltext_query(raw, require_all=False)
    if broad_q:
        row = cur.execute(
            "SELECT 1 FROM fulltext_fts WHERE fulltext_fts MATCH ? LIMIT 1",
            (broad_q,),
        ).fetchone()
        if row:
            return "broad_or"

    return "none"


def _min_gate(
    *,
    entity_terms: List[str],
    fts_branch: str,
    top1_score: float,
    score_floor: float,
) -> Tuple[bool, str]:
    """Apply the v4 two-condition min-gate to a retrieval result.

    Args:
        entity_terms: Canonical (non-location) entity terms that matched;
            used by condition B.
        fts_branch: The FTS branch that fired (see ``_detect_fts_branch``).
        top1_score: Top-1 BM25 score of the result set, or NaN/None if empty.
        score_floor: BM25 floor below which queries are declined.

    Returns (passed, reason) where reason is one of:
        "pass"
        "pass_phrase_match"             — phrase branch fired; floor bypassed (see below)
        "no_results"                    — retrieval returned nothing
        "score_below_floor"             — top-1 BM25 < floor (catches cadmium/schooner/swallow probes
                                          and very low-confidence in-domain queries)
        "no_entities_broad_or_only"     — no canonical entities matched AND the FTS branch fell
                                          through to broad-OR (catches the Switzerland-style probes
                                          where score is high but match is incidental)

    Why phrase-branch matches bypass the score floor
    ------------------------------------------------
    SQLite FTS5's bm25() ranks phrase queries as a single position-constrained
    token, whereas strict_AND on the same words accumulates a per-token score
    per word — so the SAME chunk routinely scores ~3x lower on the phrase
    branch than on strict_AND (e.g. phrase=11 vs strict_AND=33). The score
    floor was calibrated against strict_AND/broad_OR retrieval, and applying
    it unmodified to phrase-branch results spuriously declines queries whose
    verbatim text appears in a small number of chunks.

    A phrase match is *the strongest possible* in-corpus signal — by
    construction the off-corpus probes the floor was designed to catch
    ("cadmium schooner swallow") never match as phrases. So when the FTS
    branch that fired is "phrase", we treat the match itself as sufficient
    evidence and bypass the floor. The "no_entities_broad_or_only" guard
    still applies on the other branches.
    """
    if top1_score is None or (
        isinstance(top1_score, float) and math.isnan(top1_score)
    ):
        return False, "no_results"
    if fts_branch == "phrase":
        # Phrase-match — sufficient on its own. Skip both subsequent checks
        # since (a) the score floor is mis-calibrated for phrase BM25 and
        # (b) a verbatim phrase match is necessarily not an incidental
        # broad-OR hit.
        return True, "pass_phrase_match"
    if float(top1_score) < float(score_floor):
        return False, "score_below_floor"
    if not entity_terms and fts_branch == "broad_or":
        return False, "no_entities_broad_or_only"
    return True, "pass"


def _hybrid_search_sqlite4(
    state: RetrievalState,
    *,
    entity_terms: List[str],
    fulltext_query: str,
    top_k: int,
    require_all_entities: bool = True,
    subsets: List[str] = None,
    score_floor: float = MIN_GATE_SCORE_FLOOR,
    non_location_entity_terms: Optional[List[str]] = None,
    meta_out: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run the v4 hybrid retrieval algorithm (v2 retrieval + min-gate).

    Delegates retrieval to ``_hybrid_search_sqlite2`` and then applies the
    query-level min-gate (``_min_gate``). If the gate passes, v2's ranked
    results are returned unchanged; if it declines, ``[]`` is returned.

    Args:
        state: Shared retrieval state.
        entity_terms: All canonical entity terms — used for v2's underlying
            entity-FTS retrieval (locations included; we still want them to
            *match* docs).
        fulltext_query: Raw query text for the full-text branch.
        top_k: Number of results to return.
        require_all_entities: If True the entity MATCH joins terms with AND.
        subsets: Optional list of subset names to restrict results to.
        score_floor: BM25 floor passed to the min-gate.
        non_location_entity_terms: Filtered list excluding location-typed
            entities, used by the gate's condition-B check ("no entities +
            broad-OR only"). When ``None`` (e.g. legacy callers), defaults to
            ``entity_terms`` and the gate behaves like the un-refined version.
        meta_out: Optional dict mutated in place with gate decision metadata
            (``gate_decision``, ``gate_reason``, ``top1_score``,
            ``n_canonical_entities``, ``n_non_location_entities``,
            ``fts_branch_used``, ``score_floor``, ``pre_gate_n_results``).

    Returns:
        v2's list of result dicts if the gate passes, otherwise ``[]``.
    """
    results = _hybrid_search_sqlite2(
        state,
        entity_terms=entity_terms,
        fulltext_query=fulltext_query,
        top_k=top_k,
        require_all_entities=require_all_entities,
        subsets=subsets,
    )

    # Top-1 BM25 score from v2's merged ranking (NaN if no results).
    top1: float = float("nan")
    if results:
        try:
            v = results[0].get("score_bm25")
            if v is None:
                v = results[0].get("score")
            top1 = float(v) if v is not None else float("nan")
        except (TypeError, ValueError):
            top1 = float("nan")

    fts_branch = _detect_fts_branch(state, fulltext_query)
    n_ent = len(entity_terms or [])
    # Locations are a weak entity signal for declassified/conspiracy corpora —
    # "Switzerland" or "Cuba" mentioned alone shouldn't keep an off-topic query
    # alive when broad-OR is the only branch that matched. The gate uses the
    # non-location count so condition B fires for location-only queries.
    nlet = (
        non_location_entity_terms
        if non_location_entity_terms is not None
        else entity_terms
    )
    n_non_loc = len(nlet or [])

    passed, reason = _min_gate(
        entity_terms=nlet,
        fts_branch=fts_branch,
        top1_score=top1,
        score_floor=score_floor,
    )

    if meta_out is not None:
        meta_out["gate_decision"] = "pass" if passed else "decline"
        meta_out["gate_reason"] = reason
        meta_out["top1_score"] = None if math.isnan(top1) else top1
        meta_out["n_canonical_entities"] = n_ent
        meta_out["n_non_location_entities"] = n_non_loc
        meta_out["fts_branch_used"] = fts_branch
        meta_out["score_floor"] = float(score_floor)
        meta_out["pre_gate_n_results"] = len(results)

    if not passed:
        top1_str = "nan" if math.isnan(top1) else f"{top1:.2f}"
        rag_logger.info(
            "v4 min_gate DECLINED: reason=%s top1=%s n_ent=%d n_non_loc=%d fts_branch=%s pre_n=%d",
            reason,
            top1_str,
            n_ent,
            n_non_loc,
            fts_branch,
            len(results),
        )
        return []

    rag_logger.info(
        "v4 min_gate PASS: top1=%.2f n_ent=%d n_non_loc=%d fts_branch=%s n=%d",
        top1,
        n_ent,
        n_non_loc,
        fts_branch,
        len(results),
    )
    return results


# ------------------ V5: V4 + TOPIC-MATCH BOOST ------------------

# Knobs (env-overridable). V5 stays close to V4 by default; the topic
# boost is tunable independently so it can be dialed back without
# touching the gate calibration.
V5_TOPIC_BOOST_ALPHA = float(os.getenv("V5_TOPIC_BOOST_ALPHA", "0.25"))
V5_CANDIDATE_MULTIPLIER = int(os.getenv("V5_CANDIDATE_MULTIPLIER", "3"))
V5_FALLBACK_SEED_K = int(os.getenv("V5_FALLBACK_SEED_K", "5"))
V5_FALLBACK_TOPIC_K = int(os.getenv("V5_FALLBACK_TOPIC_K", "3"))

# Cache: db_path → {lemma: frozenset(topic_id)} so we lemmatize topic
# names exactly once per process per DB. The DB rarely changes mid-run;
# if it does, the cache TTL is the process lifetime which matches
# every other state-derived cache in this module.
_V5_TOPIC_VOCAB_CACHE: Dict[str, Dict[str, frozenset]] = {}

# Lemma-level stop words. Topic names like 'food_security_and_global_supply'
# tokenize to ['food','security','and','global','supply']; the 'and' would
# match every query that uses 'and' and pollute the index.
_V5_STOPWORDS = {
    "and",
    "or",
    "of",
    "for",
    "to",
    "the",
    "a",
    "an",
    "is",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "as",
}


def _v5_build_topic_lemma_index(state: RetrievalState) -> Dict[str, frozenset]:
    """Build (or return cached) lemma→topic_ids index using spaCy.

    Tokenizes each topics row's domain + topic_name on ``_`` and ``/``,
    runs each token through ``state.nlp`` to get the lemma, and maps that
    lemma to the topic_id. Multi-domain duplicate names produce sets of
    multiple topic_ids — both fire when a query lemma matches.

    Args:
        state: Shared retrieval state; the index is cached per DB path in
            ``_V5_TOPIC_VOCAB_CACHE``.

    Returns:
        A dict mapping each lemma (and its raw token form) to a frozenset of
        topic_ids whose name/domain contains it.
    """
    key = str(state.sqlite_db_path)
    cached = _V5_TOPIC_VOCAB_CACHE.get(key)
    if cached is not None:
        return cached

    index: Dict[str, set] = {}
    conn = _get_sqlite_conn(state)
    rows = conn.execute(
        "SELECT topic_id, domain, topic_name FROM topics"
    ).fetchall()
    for row in rows:
        topic_id = int(row["topic_id"])
        # Combine domain + name, split on _ / and whitespace.
        raw = f"{row['domain'] or ''} {row['topic_name'] or ''}"
        tokens = re.split(r"[_/\s]+", raw.lower())
        # Lemmatize each token through spaCy. Single tokens are cheap;
        # we accept up to ~5 tokens per topic × 172 topics = ~860 calls.
        for tok in tokens:
            if not tok or tok in _V5_STOPWORDS or len(tok) < 3:
                continue
            try:
                lemma = state.nlp(tok)[0].lemma_.lower()
            except Exception:
                lemma = tok
            index.setdefault(lemma, set()).add(topic_id)
            # Also index the raw token so plural/spelling variants both hit
            if lemma != tok:
                index.setdefault(tok, set()).add(topic_id)

    frozen = {k: frozenset(v) for k, v in index.items()}
    _V5_TOPIC_VOCAB_CACHE[key] = frozen
    rag_logger.info(
        "v5 topic lemma index built: %d lemmas → %d topics",
        len(frozen),
        len(rows),
    )
    return frozen


def _v5_infer_query_topics_spacy(query: str, state: RetrievalState) -> set:
    """Infer query topic_ids by spaCy lemma overlap with topic names.

    Lemmatizes the query's content tokens (NOUN/PROPN/ADJ/VERB, non-stop,
    length >= 3) and looks each lemma up in the topic lemma index.

    Args:
        query: The raw query string.
        state: Shared retrieval state.

    Returns:
        A set of matching topic_ids; empty set on no query or no match.
    """
    if not query:
        return set()
    index = _v5_build_topic_lemma_index(state)
    if not index:
        return set()
    doc = state.nlp(query)
    candidate_lemmas = {
        t.lemma_.lower()
        for t in doc
        if t.pos_ in {"NOUN", "PROPN", "ADJ", "VERB"}
        and not t.is_stop
        and len(t.lemma_) >= 3
    }
    hits: set = set()
    for lemma in candidate_lemmas:
        hits.update(index.get(lemma, ()))
    return hits


def _v5_infer_query_topics_cooccurrence(
    initial_results: List[Dict[str, Any]],
    state: RetrievalState,
    *,
    seed_k: int = V5_FALLBACK_SEED_K,
    top_topics: int = V5_FALLBACK_TOPIC_K,
) -> set:
    """Infer query topic_ids by co-occurrence among the top initial results.

    Fallback used only when the spaCy lemma pass yields nothing: takes the
    topics of the top ``seed_k`` initial results and returns the most common
    ``top_topics`` topic_ids.

    Args:
        initial_results: The initial (pre-boost) result list.
        state: Shared retrieval state.
        seed_k: Number of leading results to seed from.
        top_topics: Number of most-common topic_ids to return.

    Returns:
        A set of up to ``top_topics`` topic_ids; empty if there are no seeds.
    """
    seeds = [
        r.get("lookup_id")
        for r in (initial_results or [])[:seed_k]
        if r.get("lookup_id") is not None
    ]
    if not seeds:
        return set()
    placeholders = ",".join("?" * len(seeds))
    conn = _get_sqlite_conn(state)
    rows = conn.execute(
        f"SELECT topic_id FROM chunk_topics WHERE chunk_lookup_id IN ({placeholders})",
        seeds,
    ).fetchall()
    counts: Counter = Counter(int(r["topic_id"]) for r in rows)
    return {tid for tid, _ in counts.most_common(top_topics)}


def _v5_fetch_chunk_topics(
    lookup_ids: List[int],
    state: RetrievalState,
) -> Dict[int, set]:
    """Bulk-fetch topic_ids for each candidate chunk.

    Chunks with no ``chunk_topics`` row get an empty set, which means a zero
    boost later — exactly what we want when topics are missing for new
    subsets (e.g. PEERS Substack ingested without ``--with-topics``).

    Args:
        lookup_ids: Chunk ``lookup_id`` values to fetch topics for.
        state: Shared retrieval state.

    Returns:
        A dict mapping every input ``lookup_id`` to its set of topic_ids
        (empty set if the chunk has no topics); ``{}`` if ``lookup_ids`` empty.
    """
    if not lookup_ids:
        return {}
    placeholders = ",".join("?" * len(lookup_ids))
    conn = _get_sqlite_conn(state)
    rows = conn.execute(
        f"SELECT chunk_lookup_id, topic_id FROM chunk_topics "
        f"WHERE chunk_lookup_id IN ({placeholders})",
        lookup_ids,
    ).fetchall()
    out: Dict[int, set] = {lid: set() for lid in lookup_ids}
    for r in rows:
        out[int(r["chunk_lookup_id"])].add(int(r["topic_id"]))
    return out


def _hybrid_search_sqlite5(
    state: RetrievalState,
    *,
    entity_terms: List[str],
    fulltext_query: str,
    top_k: int,
    require_all_entities: bool = True,
    subsets: List[str] = None,
    score_floor: float = MIN_GATE_SCORE_FLOOR,
    non_location_entity_terms: Optional[List[str]] = None,
    meta_out: Optional[Dict[str, Any]] = None,
    topic_boost_alpha: float = V5_TOPIC_BOOST_ALPHA,
    candidate_multiplier: int = V5_CANDIDATE_MULTIPLIER,
) -> List[Dict[str, Any]]:
    """
    v5 = v4's gate + a topic-match multiplicative boost.

    Pipeline:
        1) Pull an expanded candidate pool from v2 (top_k * candidate_multiplier).
        2) Infer the query's topic_ids:
             a) spaCy lemma match against precomputed topic-name vocab
             b) fallback: most-common topic_ids among initial top-N results
        3) Fetch chunk_topics for every candidate in one batched query.
        4) Boost: score' = score_bm25 * (1 + alpha * overlap_frac), where
           overlap_frac = |chunk_topics ∩ query_topics| / max(1, |chunk_topics|).
        5) Re-sort by boosted score, take top_k.
        6) Apply v4's min-gate to the PRE-boost top-1 score, so the score
           floor calibration is unchanged from v4.

    Degrades gracefully:
      * No topic_ids match the query  → all overlap_frac == 0 → identical to v4
      * Candidate chunks have no chunk_topics rows → same as above
      * Topics table empty → spaCy index is empty, fallback returns empty,
        and behavior collapses to v4 (which collapses to v2's results).

    Args:
        state: Shared retrieval state.
        entity_terms: All canonical entity terms for v2's entity branch.
        fulltext_query: Raw query text for the full-text branch.
        top_k: Number of results to return.
        require_all_entities: If True the entity MATCH joins terms with AND.
        subsets: Optional list of subset names to restrict results to.
        score_floor: BM25 floor passed to the min-gate.
        non_location_entity_terms: Filtered list excluding location-typed
            entities, used by the gate's condition-B check; defaults to
            ``entity_terms`` when ``None``.
        meta_out: Optional dict mutated in place with v4 gate metadata plus
            v5 boost diagnostics (``topic_inference_source``,
            ``n_query_topics``, ``query_topic_ids``, ``n_candidates_boosted``,
            ``boost_alpha``).
        topic_boost_alpha: Multiplicative boost coefficient for topic overlap.
        candidate_multiplier: Factor by which the candidate pool exceeds
            ``top_k`` before re-ranking.

    Returns:
        A list of up to ``top_k`` result dicts (each carrying ``topic_overlap``,
        ``topic_boost`` and ``topic_score`` keys) if the gate passes,
        otherwise ``[]``.
    """
    # Expanded retrieval pool so chunks ranked just-below top_k can climb
    # when their topics match the query.
    pool_k = max(int(top_k) * max(1, int(candidate_multiplier)), int(top_k))
    initial = _hybrid_search_sqlite2(
        state,
        entity_terms=entity_terms,
        fulltext_query=fulltext_query,
        top_k=pool_k,
        require_all_entities=require_all_entities,
        subsets=subsets,
    )

    # Capture pre-boost top1 BEFORE we reorder. The gate threshold is
    # tuned on v4's BM25 scores so we want to gate on the same signal.
    pre_boost_top1: float = float("nan")
    if initial:
        try:
            v = initial[0].get("score_bm25")
            if v is None:
                v = initial[0].get("score")
            pre_boost_top1 = float(v) if v is not None else float("nan")
        except (TypeError, ValueError):
            pre_boost_top1 = float("nan")

    # ---- Topic inference ------------------------------------------------- #
    query_topics = _v5_infer_query_topics_spacy(fulltext_query, state)
    topic_source = "spacy" if query_topics else "none"
    if not query_topics:
        query_topics = _v5_infer_query_topics_cooccurrence(initial, state)
        if query_topics:
            topic_source = "cooccurrence"

    # ---- Per-chunk topic fetch + boost ----------------------------------- #
    n_boosted = 0
    if query_topics and initial:
        lookup_ids = [
            r["lookup_id"] for r in initial if r.get("lookup_id") is not None
        ]
        chunk_topics_map = _v5_fetch_chunk_topics(lookup_ids, state)
        for r in initial:
            lid = r.get("lookup_id")
            ctopics = chunk_topics_map.get(lid) if lid is not None else None
            if not ctopics:
                r["topic_overlap"] = 0
                r["topic_boost"] = 1.0
                r["topic_score"] = float(r.get("score_bm25") or 0.0)
                continue
            overlap = ctopics & query_topics
            overlap_frac = len(overlap) / max(1, len(ctopics))
            boost = 1.0 + float(topic_boost_alpha) * float(overlap_frac)
            r["topic_overlap"] = len(overlap)
            r["topic_overlap_frac"] = overlap_frac
            r["topic_boost"] = boost
            base = float(r.get("score_bm25") or 0.0)
            r["topic_score"] = base * boost
            if overlap:
                n_boosted += 1
        initial.sort(key=lambda d: d.get("topic_score", 0.0), reverse=True)
    else:
        # Stamp a neutral boost so callers can rely on the keys being present.
        for r in initial:
            r.setdefault("topic_overlap", 0)
            r.setdefault("topic_boost", 1.0)
            r.setdefault("topic_score", float(r.get("score_bm25") or 0.0))

    final = initial[: int(top_k)]

    # ---- Min-gate (PRE-boost top-1 only) --------------------------------- #
    fts_branch = _detect_fts_branch(state, fulltext_query)
    n_ent = len(entity_terms or [])
    nlet = (
        non_location_entity_terms
        if non_location_entity_terms is not None
        else entity_terms
    )
    n_non_loc = len(nlet or [])
    passed, reason = _min_gate(
        entity_terms=nlet,
        fts_branch=fts_branch,
        top1_score=pre_boost_top1,
        score_floor=score_floor,
    )

    if meta_out is not None:
        meta_out["gate_decision"] = "pass" if passed else "decline"
        meta_out["gate_reason"] = reason
        meta_out["top1_score"] = (
            None if math.isnan(pre_boost_top1) else pre_boost_top1
        )
        meta_out["n_canonical_entities"] = n_ent
        meta_out["n_non_location_entities"] = n_non_loc
        meta_out["fts_branch_used"] = fts_branch
        meta_out["score_floor"] = float(score_floor)
        meta_out["pre_gate_n_results"] = len(initial)
        # v5-specific:
        meta_out["topic_inference_source"] = topic_source
        meta_out["n_query_topics"] = len(query_topics)
        meta_out["query_topic_ids"] = sorted(query_topics)
        meta_out["n_candidates_boosted"] = n_boosted
        meta_out["boost_alpha"] = float(topic_boost_alpha)

    if not passed:
        top1_str = (
            "nan" if math.isnan(pre_boost_top1) else f"{pre_boost_top1:.2f}"
        )
        rag_logger.info(
            "v5 min_gate DECLINED: reason=%s top1=%s n_ent=%d n_non_loc=%d "
            "fts_branch=%s pre_n=%d query_topics=%d boosted=%d",
            reason,
            top1_str,
            n_ent,
            n_non_loc,
            fts_branch,
            len(initial),
            len(query_topics),
            n_boosted,
        )
        return []

    rag_logger.info(
        "v5 min_gate PASS: top1=%.2f n_ent=%d n_non_loc=%d fts_branch=%s "
        "topic_src=%s query_topics=%d boosted=%d n=%d",
        pre_boost_top1,
        n_ent,
        n_non_loc,
        fts_branch,
        topic_source,
        len(query_topics),
        n_boosted,
        len(final),
    )
    return final


# ------------------ SEARCH REFERENCES SKELETON ------------------


async def search_references(
    state: RetrievalState,
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    verbose: bool = False,
    entity_source_query: Optional[str] = None,
    fulltext_query: Optional[str] = None,
    subsets: List[str] = None,
    rag_algo_choice: int = 5,  # v5 is the production default (topic-boosted v4 + min-gate)
    **_unused,
) -> Dict[str, Any]:
    """Public retrieval entry point: dispatch to a hybrid search algorithm.

    Extracts canonical entity terms (typed, so locations can be excluded from
    gating), then dispatches to one of the five ``_hybrid_search_sqlite``
    variants by ``rag_algo_choice``. Variants 4 and 5 also attach min-gate
    metadata to the returned dict.

    Args:
        state: Shared retrieval state.
        query: The user's query string.
        top_k: Number of results to return.
        verbose: Reserved verbosity flag (currently unused).
        entity_source_query: Optional alternate text to extract entities from;
            defaults to ``query``.
        fulltext_query: Optional alternate text for the full-text branch;
            defaults to ``query``.
        subsets: Optional list of subset names to restrict results to.
        rag_algo_choice: Which algorithm to run (1-5); unknown values fall
            back to v1. Defaults to 5 (the production default).
        **_unused: Ignored extra keyword arguments.

    Returns:
        A dict with keys ``query``, ``num_results``, ``results`` (the list of
        result dicts) and ``message``. For ``rag_algo_choice`` 4 or 5 the
        min-gate metadata keys are merged in as well. Empty queries return a
        zero-result dict with ``message`` set to ``"Empty query."``.
    """
    q = (query or "").strip()
    if not q:
        return {
            "query": query,
            "num_results": 0,
            "results": [],
            "message": "Empty query.",
        }

    entity_source = (entity_source_query or q).strip()
    fulltext_source = (fulltext_query or q).strip()

    typed_entity_terms = extract_canonical_entity_terms_typed(
        entity_source, state
    )
    entity_terms = [t for t, _ in typed_entity_terms]
    non_location_entity_terms = [
        t for t, cat in typed_entity_terms if cat != "locations"
    ]
    rag_logger.info(
        f"search_references entity_terms={entity_terms} (non_loc={non_location_entity_terms}), "
        f"subsets={subsets}, rag_algo_choice={rag_algo_choice}"
    )

    gate_meta: Optional[Dict[str, Any]] = None  # populated only for v4

    match rag_algo_choice:
        case 1:
            results = _hybrid_search_sqlite(
                state,
                entity_terms=entity_terms,
                fulltext_query=fulltext_source,
                top_k=int(top_k),
                require_all_entities=True if entity_terms else False,
                subsets=subsets,
            )
        case 2:
            results = _hybrid_search_sqlite2(
                state,
                entity_terms=entity_terms,
                fulltext_query=fulltext_source,
                top_k=int(top_k),
                require_all_entities=True if entity_terms else False,
                subsets=subsets,
            )
        case 3:
            results = _hybrid_search_sqlite3(
                state,
                entity_terms=entity_terms,
                fulltext_query=fulltext_source,
                top_k=int(top_k),
                require_all_entities=True if entity_terms else False,
                subsets=subsets,
            )
        case 4:
            gate_meta = {}
            results = _hybrid_search_sqlite4(
                state,
                entity_terms=entity_terms,
                non_location_entity_terms=non_location_entity_terms,
                fulltext_query=fulltext_source,
                top_k=int(top_k),
                require_all_entities=True if entity_terms else False,
                subsets=subsets,
                meta_out=gate_meta,
            )
        case 5:
            gate_meta = {}
            results = _hybrid_search_sqlite5(
                state,
                entity_terms=entity_terms,
                non_location_entity_terms=non_location_entity_terms,
                fulltext_query=fulltext_source,
                top_k=int(top_k),
                require_all_entities=True if entity_terms else False,
                subsets=subsets,
                meta_out=gate_meta,
            )
        case _:
            results = _hybrid_search_sqlite(
                state,
                entity_terms=entity_terms,
                fulltext_query=fulltext_source,
                top_k=int(top_k),
                require_all_entities=True if entity_terms else False,
                subsets=subsets,
            )

    out: Dict[str, Any] = {
        "query": query,
        "num_results": len(results),
        "results": results,
        "message": f"Found {len(results)} result(s).",
    }
    # v4 attaches gate metadata; older variants leave these absent.
    if gate_meta:
        out.update(gate_meta)
    return out
