"""
rag_controller.py (SQLite hybrid edition)
- Uses a hybrid SQLite FTS5 backend:
  - entity/topic search in one FTS table
  - title + fulltext search in another FTS table
- Preserves the existing Flask-facing interface:
  - boot()
  - search_references(...)
  - ask(...)
  - ask_model_only(...)
- Keeps queueing / model orchestration helpers
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import spacy
from dataclasses_json import dataclass_json
import string

import logging_config
import model_adapters
import rag_cleaner
import utils

rag_logger = logging_config.get_logger("rag")

TRINEDAY_TOKEN = "Trine Day Publishing"
COPYRIGHT_BLOCK_MSG = "This text protected by copyright."

# ------------------ MODEL ADAPTERS ------------------
# Two model adapters enter, one model adapter leaves.
hf_llm_model = model_adapters.LLMFactory.create("hf")
deep_infra_llm_model = model_adapters.LLMFactory.create("deepinfra")

llm_model = hf_llm_model
llm_models = [hf_llm_model, deep_infra_llm_model]


def get_model_type(type: str) -> model_adapters.LLMStrategy:
    global llm_model, hf_llm_model, deep_infra_llm_model

    if not model_adapters.is_valid_model_type(type):
        raise ValueError(f"Invalid model type: {type}")

    if type == "hf":
        return hf_llm_model
    elif type == "deepinfra":
        return deep_infra_llm_model
    elif type == "default":
        return llm_model
    else:
        raise ValueError(f"Unknown model type: {type}")


# ------------------ CONFIG ------------------

MAX_QUESTION_WORDS = 400
DEFAULT_TOP_K = int(os.getenv("TRINEDAY_TOP_K", "10"))
ENABLE_MIN_GATING = False

RETRIEVAL_BACKEND = os.getenv("RETRIEVAL_BACKEND", "sqlite_hybrid")
HYBRID_DB_PATH = Path(os.getenv("HYBRID_DB_PATH", "./data/gamma_master_hybrid_fts.db"))
ENTITY_CANON_MAP_PATH = Path(
    os.getenv("ENTITY_CANON_MAP_PATH", "./data/entity_query_normalization_map.flat.json")
)
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

HYBRID_ENTITY_WEIGHT = float(os.getenv("HYBRID_ENTITY_WEIGHT", "4.0"))
HYBRID_FULLTEXT_WEIGHT = float(os.getenv("HYBRID_FULLTEXT_WEIGHT", "1.0"))
HYBRID_ENTITY_LIMIT = int(os.getenv("HYBRID_ENTITY_LIMIT", "200"))
HYBRID_FULLTEXT_LIMIT = int(os.getenv("HYBRID_FULLTEXT_LIMIT", "200"))

STOPWORDS = {
    "what", "about", "the", "a", "an", "of", "to", "for", "in", "on", "at",
    "is", "was", "are", "were", "did", "does", "do", "and", "or", "but",
    "with", "from", "by", "as", "it", "this", "that", "these", "those"
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

_STRIP_PHRASES_RE = re.compile(r"(click here|more along these lines|about us)", re.IGNORECASE)
_STOP_MARKERS = ["</s>", "<|end|>", "<|eot_id|>"]
_URL_RE = re.compile(r"(https?://[^\s<]+)")
_SENT_RE = re.compile(r"[^.!?]+[.!?]*")


# ------------------ STATE ------------------

@dataclass
class RetrievalState:
    backend: str
    sqlite_db_path: Path
    flat_lookup: Dict[str, str]
    nlp: Any
    _thread_local: Any


@dataclass
class QueuedJob:
    user_id: str
    job_id: str
    model_type: str
    prompt: str


@dataclass_json
@dataclass
class QueuedResponse:
    ok: bool
    error: str
    detail: str
    job_id: str
    user_id: str
    prompt: str
    reply: str
    references: str


# ------------------ SQLITE / BOOT ------------------

def _get_sqlite_conn(state: RetrievalState) -> sqlite3.Connection:
    """
    One SQLite connection per thread.
    Because sharing one connection across threads is how goblins are born.
    """
    conn = getattr(state._thread_local, "sqlite_conn", None)
    if conn is None:
        conn = sqlite3.connect(str(state.sqlite_db_path))
        conn.row_factory = sqlite3.Row
        state._thread_local.sqlite_conn = conn
    return conn


def boot() -> RetrievalState:
    rag_logger.info("Booting retrieval...")

    if RETRIEVAL_BACKEND != "sqlite_hybrid":
        raise RuntimeError(f"Unsupported RETRIEVAL_BACKEND: {RETRIEVAL_BACKEND}")

    if not HYBRID_DB_PATH.exists():
        raise FileNotFoundError(f"Missing hybrid SQLite DB: {HYBRID_DB_PATH}")

    if not ENTITY_CANON_MAP_PATH.exists():
        raise FileNotFoundError(f"Missing flat canonical map: {ENTITY_CANON_MAP_PATH}")

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
    s = str(s).strip().lower()
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = re.sub(r"'s\b", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonical_to_entity_term(s: str) -> str:
    s = normalize_entity_key(s)
    s = s.replace("&", "and")
    s = s.replace("/", "_")
    s = s.replace("-", "_")
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _fts_entity_surface(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("_", " ")
    s = s.replace("/", " ")
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize_fulltext_query(raw_query: str):
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
    toks = tokenize_fulltext_query(raw_query)
    if not toks:
        return ""
    joiner = " AND " if require_all else " OR "
    return joiner.join(toks)


def build_entity_match_query(entity_terms: List[str], require_all: bool = True) -> Optional[str]:
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


def extract_canonical_entity_terms(query: str, state: RetrievalState) -> List[str]:
    doc = state.nlp(query)
    seen = set()
    out: List[str] = []

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
            out.append(entity_term)
            seen.add(entity_term)

    return out


# ------------------ RETRIEVAL TEXT CLEANING ------------------

def truncate_question(q: str) -> Tuple[str, bool]:
    words = (q or "").split()
    if len(words) <= MAX_QUESTION_WORDS:
        return (q or "").strip(), False
    return " ".join(words[:MAX_QUESTION_WORDS]), True


def _strip_phrases(s: str) -> str:
    if not s:
        return ""
    return _STRIP_PHRASES_RE.sub(" ", str(s))


def _strip_on_literal_stops(text: str, stops=None) -> str:
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
    return s[:m.start()].rstrip()


def clean_retrieval_text(text: str) -> str:
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
_BROKEN_URL_DASH_RE = re.compile(r"\b[a-z0-9]+\s*-\s*[a-z0-9]+\s*-\s*[a-z0-9]+", re.I)
_BIBLIO_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_AUTHOR_QUOTE_RE = re.compile(r'"[^"]{8,}"')
_MULTI_SEMI_RE = re.compile(r";")
_MULTI_SLASH_RE = re.compile(r"/")
_REPEAT_DASH_RE = re.compile(r"(?:\s-\s){3,}")
_REFERENCE_LINE_RE = re.compile(r"\[\d{1,3}\].{0,140}?(?:https?://|www\.)", re.I)

def _punct_stats(text: str) -> dict:
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
    """
    title_stats = _punct_stats(title)
    text_stats = _punct_stats(text[:1600])

    # Extremely obvious garbage
    if text_stats["max_punct_run"] >= 8:
        return True

    if text_stats["reference_line_count"] >= 2:
        return True

    if text_stats["bracket_citation_count"] >= 4 and text_stats["urlish_count"] >= 2:
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

    if text_stats["dash_ratio"] > 0.03 and text_stats["broken_url_dash_count"] >= 2:
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
    return _URL_RE.sub(
        r'<a href="\1" target="_blank" rel="noopener noreferrer">\1</a>',
        html_text,
    )


def _truncate_words(text: str, max_words: int) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return text or ""
    return " ".join(words[:max_words]) + "…"


def _snippet_html_from_text(text: str, max_words: int = 600) -> str:
    import html as _html

    t = _truncate_words(text or "", max_words)
    t = _html.escape(t).replace("\n", "<br>")
    return _linkify_plain_urls(t)


def _sentence_split(text: str) -> List[str]:
    s = re.sub(r"\s+", " ", text or "").strip()
    if not s:
        return []
    return [m.group(0).strip() for m in _SENT_RE.finditer(s) if m.group(0).strip()]


def _to_term_set(query: str) -> set[str]:
    q = (query or "").lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    return {w for w in q.split() if len(w) > 2}


def _tokenize_simple(t: str) -> List[str]:
    t = (t or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return [w for w in t.split() if w]


def _trigrams(tokens: List[str]) -> List[str]:
    return [" ".join(tokens[i:i+3]) for i in range(len(tokens) - 2)]


def _fts_positive_score(bm25_score: float) -> float:
    # FTS5 bm25 likes to return negative values.
    # We convert it into a polite positive score instead of pretending all scores are identical.
    return 1.0 / (1.0 + abs(float(bm25_score)))


def _sqlite_row_to_result(
    row: sqlite3.Row,
    hybrid_score: float,
    entity_score: float,
    fulltext_score: float,
) -> Dict[str, Any]:
    txt = str(row["fulltext_text"] or "")
    src = str(row["source_url"] or "").strip()
    subset = str(row["subset_name"] or "").strip()

    return {
        "row_id": row["chunk_id"] or row["lookup_id"],
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
            protected = "This text is protected by copyright and cannot be displayed."
            new_doc["text"] = protected
            new_doc["snippet"] = protected
            new_doc["snippet_html"] = protected

        cleaned.append(new_doc)

    return cleaned

# ------------------ HYBRID SQLITE SEARCH ------------------

def _entity_positions(entity_term: str, text: str) -> List[int]:
    surf = re.escape(entity_term.replace("_", " "))
    return [m.start() for m in re.finditer(rf"\b{surf}\b", str(text or ""), flags=re.I)]


def _has_entity_proximity_match(entity_terms: List[str], text: str, max_chars: int = 700) -> bool:
    """
    Require at least 2 distinct entity terms to appear near each other.
    This helps block "same giant chunk, unrelated subtopics" junk.
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
    return tokenize_fulltext_query(raw_query)


def _build_phrase_query(raw_query: str) -> str:
    """
    Build an exact phrase query for longer pasted excerpts.
    Returns empty string for very short queries.
    """
    q = str(raw_query or "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    if len(q.split()) < 5:
        return ""
    return f'"{q}"'


def _keyword_overlap_bonus(raw_query: str, title: str, text: str) -> float:
    """
    Bonus for literal keyword overlap in title + early text.
    Helps excerpt search and named-term queries.
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
                phrase = " ".join(words[i:i+span])
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
) -> List[Dict[str, Any]]:
    conn = _get_sqlite_conn(state)
    cur = conn.cursor()

    entity_query = build_entity_match_query(
        entity_terms,
        require_all=require_all_entities,
    )

    raw_ft_query = str(fulltext_query or "").strip()
    phrase_fulltext_query = _build_phrase_query(raw_ft_query)
    strict_fulltext_query = build_fulltext_query(raw_ft_query, require_all=True)
    broad_fulltext_query = build_fulltext_query(raw_ft_query, require_all=False)

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
        score = _fts_positive_score(row["bm25_score"])
        merged.setdefault(
            lookup_id,
            {
                "entity_score": 0.0,
                "fulltext_score": 0.0,
            },
        )
        merged[lookup_id]["entity_score"] = max(
            merged[lookup_id]["entity_score"],
            score,
        )

    for row in fulltext_rows:
        lookup_id = int(row["lookup_id"])
        score = _fts_positive_score(row["bm25_score"])
        merged.setdefault(
            lookup_id,
            {
                "entity_score": 0.0,
                "fulltext_score": 0.0,
            },
        )
        merged[lookup_id]["fulltext_score"] = max(
            merged[lookup_id]["fulltext_score"],
            score,
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

    meta_rows = list(
        cur.execute(
            f"""
            SELECT lookup_id, chunk_id, title, subset_name, domain, fulltext_text, source_url
            FROM chunks
            WHERE lookup_id IN ({placeholders})
            """,
            ids,
        )
    )
    meta_map = {int(r["lookup_id"]): r for r in meta_rows}

    # ----------------------------
    # 5) Post-filter + score bonuses
    # ----------------------------
    rescored = []
    dropped_noisy = 0
    dropped_proximity = 0
    dropped_anchor = 0

    for lookup_id, hybrid_score, entity_score, fulltext_score in ranked:
        row = meta_map.get(lookup_id)
        if row is None:
            continue

        title = str(row["title"] or "")
        fulltext_text = str(row["fulltext_text"] or "")

        if _looks_too_punct_noisy(title, fulltext_text):
            dropped_noisy += 1
            continue

        if len(entity_terms) >= 2:
            if not _has_entity_proximity_match(entity_terms, fulltext_text, max_chars=700):
                dropped_proximity += 1
                continue

        if entity_terms:
            if not _has_early_anchor(entity_terms, title, fulltext_text):
                dropped_anchor += 1
                continue

        keyword_bonus = _keyword_overlap_bonus(raw_ft_query, title, fulltext_text)
        phrase_bonus = _exact_phrase_bonus(raw_ft_query, title, fulltext_text)

        adjusted_hybrid_score = (
            hybrid_score
            - _generic_title_penalty(title)
            + keyword_bonus
            + phrase_bonus
        )

        rescored.append(
            (
                lookup_id,
                adjusted_hybrid_score,
                entity_score,
                fulltext_score,
                keyword_bonus,
                phrase_bonus,
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
    ) in rescored:
        row = meta_map.get(lookup_id)
        if row is None:
            continue

        result = _sqlite_row_to_result(
            row,
            adjusted_hybrid_score,
            entity_score,
            fulltext_score,
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


async def search_references(
    state: RetrievalState,
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    verbose: bool = False,
    entity_source_query: Optional[str] = None,
    fulltext_query: Optional[str] = None,
    **_unused,
) -> Dict[str, Any]:
    q = (query or "").strip()
    if not q:
        return {"query": query, "num_results": 0, "results": [], "message": "Empty query."}

    entity_source = (entity_source_query or q).strip()
    fulltext_source = (fulltext_query or q).strip()

    entity_terms = extract_canonical_entity_terms(entity_source, state)
    rag_logger.info("search_references entity_terms=%s", entity_terms)

    results = _hybrid_search_sqlite(
        state,
        entity_terms=entity_terms,
        fulltext_query=fulltext_source,
        top_k=int(top_k),
        require_all_entities=True if entity_terms else False,
    )

    return {
        "query": query,
        "num_results": len(results),
        "results": results,
        "message": f"Found {len(results)} result(s).",
    }


# ------------------ CONTEXT BUILD ------------------

def build_context(
    docs: List[Dict[str, Any]],
    query: str,
    *,
    context_k: int = 5,
    max_snips_per_doc: int = 5,
    max_sent_per_doc: int = 50,
) -> str:
    terms = _to_term_set(query)
    global_tris: set[str] = set()
    blocks: List[str] = []

    picked = (docs or [])[: int(context_k)]

    for i, d in enumerate(picked, start=1):
        text = (d.get("text") or d.get("snippet") or "").strip()
        if not text:
            continue

        rag_logger.info(f"building context pre-text for doc {i}: {text}")
        text = rag_cleaner.clean_text_for_rag(text)
        rag_logger.info(f"building context cleaned-text for doc {i}: {text}")

        sents = _sentence_split(text)[: int(max_sent_per_doc)]
        kept: List[str] = []

        for s in sents:
            low = s.lower()
            if not any(t in low for t in terms):
                continue
            if len(kept) >= int(max_snips_per_doc):
                break

            toks = _tokenize_simple(s)
            tris = _trigrams(toks)
            introduce = any(tri not in global_tris for tri in tris)
            if introduce:
                kept.append(s.strip())
                for tri in tris:
                    global_tris.add(tri)

        if not kept:
            continue

        doc_id = d.get("row_id") or d.get("title") or ""
        url = d.get("source") or ""
        score = d.get("score_bm25") or 0.0

        block = (
            f'<doc id="{doc_id}" url="{url}" score="{float(score):.2f}">\n'
            "<snippets>\n- " + "\n- ".join(kept) + "\n</snippets>\n"
            "</doc>"
        )
        blocks.append(block)

    return "\n".join(blocks)


def build_context_improved(
    docs: List[Dict[str, Any]],
    query: str,
    *,
    context_k: int = 5,
    max_snips_per_doc: int = 5,
    max_sent_per_doc: int = 50,
    window_radius: int = 1,
) -> str:
    terms = _to_term_set(query)
    global_tris: set[str] = set()
    blocks: List[str] = []
    picked = (docs or [])[: int(context_k)]

    for i, d in enumerate(picked, start=1):
        text = (d.get("text") or d.get("snippet") or "").strip()
        if not text:
            continue

        rag_logger.info(f"building context pre-text for doc {i}: {text}")
        text = rag_cleaner.clean_text_for_rag(text)
        rag_logger.info(f"building context cleaned-text for doc {i}: {text}")

        sents = _sentence_split(text)[: int(max_sent_per_doc)]
        if not sents:
            continue

        scored_indices = []
        for idx, s in enumerate(sents):
            low = s.lower()
            sent_tokens = set(_tokenize_simple(low))
            overlap = len(terms & sent_tokens)
            if overlap == 0:
                continue
            scored_indices.append((idx, overlap))

        if not scored_indices:
            continue

        scored_indices.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in scored_indices[:max_snips_per_doc]]

        windows = []
        for idx in top_indices:
            start = max(0, idx - window_radius)
            end = min(len(sents), idx + window_radius + 1)
            windows.append((start, end))

        windows.sort()
        merged = []
        for start, end in windows:
            if not merged:
                merged.append([start, end])
            else:
                prev_start, prev_end = merged[-1]
                if start <= prev_end:
                    merged[-1][1] = max(prev_end, end)
                else:
                    merged.append([start, end])

        kept_blocks: List[str] = []
        for start, end in merged:
            window_text = " ".join(sents[start:end]).strip()
            toks = _tokenize_simple(window_text)
            tris = _trigrams(toks)

            introduce = any(tri not in global_tris for tri in tris)
            if introduce:
                kept_blocks.append(window_text)
                for tri in tris:
                    global_tris.add(tri)

        if not kept_blocks:
            continue

        doc_id = d.get("row_id") or d.get("title") or ""
        url = d.get("source") or ""
        score = d.get("score_bm25") or 0.0

        block = (
            f'<doc id="{doc_id}" url="{url}" score="{float(score):.2f}">\n'
            "<snippets>\n- " + "\n- ".join(kept_blocks) + "\n</snippets>\n"
            "</doc>"
        )
        blocks.append(block)

    return "\n".join(blocks)


# ------------------ HF ORCHESTRATION ------------------

def is_model_ready(timeout=model_adapters.MODEL_TIMEOUT_SECS) -> bool:
    return utils.do_async_to_sync(lambda: llm_model.is_model_ready(timeout=timeout))()


def send_warmup() -> bool:
    return utils.do_async_to_sync(lambda: llm_model.send_warmup())()


async def is_model_ready_deprecated(timeout=model_adapters.MODEL_TIMEOUT_SECS) -> bool:
    rag_logger.info("checking health...")
    try:
        r = requests.post(
            f"{model_adapters.HF_ENDPOINT_URL}",
            headers=model_adapters.HF_REQ_HEADERS,
            json=model_adapters.HF_HEALTH_PAYLOAD,
            timeout=timeout,
        )
        if r.status_code == 200:
            if r.json().get("health") == "ok":
                rag_logger.info("Model ready: Custom health response received")
                return True
            else:
                rag_logger.info("Processed response but not explicit health OK")
                return False
        elif r.status_code in (401, 403):
            rag_logger.error(f"Auth error {r.status_code}: Invalid HF_API_KEY?")
            return False
        elif r.status_code == 503:
            rag_logger.info("503: Model likely still loading (cold start)")
            return False
        else:
            rag_logger.warning(f"Unexpected status: {r.status_code} - {r.text}")
            return False
    except requests.Timeout:
        rag_logger.warning("Health check timed out (model loading?)")
        return False
    except Exception as e:
        rag_logger.warning(f"Health check failed: {e}")
        return False


async def send_warmup_deprecated() -> bool:
    payload = {
        "inputs": model_adapters.HF_WARMUP_PROMPT,
        "parameters": {
            "max_new_tokens": model_adapters.HF_WARMUP_MAX_NEW_TOKENS,
            "temperature": 0.1,
            "stop_sequences": ["\n", "Q:"],
        },
    }
    try:
        r = requests.post(
            f"{model_adapters.HF_ENDPOINT_URL}/generate",
            json=payload,
            headers=model_adapters.HF_REQ_HEADERS,
            timeout=60,
        )
        if r.status_code == 200:
            rag_logger.info(
                "Warm-up successful! Response: %s",
                r.json().get("generated_text", "")[:100],
            )
            return True
    except Exception as e:
        rag_logger.warning(f"Warm-up request failed: {e}")

    return False


def _parse_hf_text_deprecated(data) -> str:
    if isinstance(data, list) and data and isinstance(data[0], dict):
        if data[0].get("generated_text"):
            return str(data[0]["generated_text"])
    if isinstance(data, dict):
        if data.get("generated_text"):
            return str(data["generated_text"])
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0] or {}
            txt = c0.get("text") or (c0.get("message") or {}).get("content") or ""
            if txt:
                return str(txt)
    if isinstance(data, str):
        return data
    return str(data)


def _hf_generate_deprecated(prompt: str, *, temperature: float, max_new_tokens: int) -> str:
    if not model_adapters.HF_ENDPOINT_URL:
        raise RuntimeError("Missing HF_ENDPOINT_URL")
    if not model_adapters.HF_API_KEY:
        raise RuntimeError("Missing HF_API_KEY")

    headers = {
        "Authorization": f"Bearer {model_adapters.HF_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": float(temperature),
            "max_new_tokens": int(max_new_tokens),
            "return_full_text": False,
        },
    }

    last_detail = None

    for attempt in range(1, model_adapters.HF_MAX_ATTEMPTS + 1):
        r = requests.post(
            model_adapters.HF_ENDPOINT_URL,
            headers=headers,
            json=payload,
            timeout=model_adapters.HF_TIMEOUT,
        )

        if r.status_code == 503:
            try:
                j = r.json()
            except Exception:
                j = {}
            wait = int(j.get("estimated_time") or 3)
            wait = max(1, min(model_adapters.HF_MAX_WAIT_SECS, wait))
            time.sleep(wait)
            last_detail = f"503 loading; wait={wait}s; attempt={attempt}/{model_adapters.HF_MAX_ATTEMPTS}"
            continue

        if not r.ok:
            body = (r.text or "")[:2000]
            raise RuntimeError(f"HF error {r.status_code}: {body}")

        try:
            data = r.json()
        except Exception:
            raise RuntimeError(f"HF returned non-JSON: {(r.text or '')[:2000]}")

        return _parse_hf_text_deprecated(data).strip()

    raise RuntimeError(f"HF model still loading (503). Last: {last_detail or 'n/a'}")


async def ask(
    model_adaptor,
    state: RetrievalState,
    question: str,
    *,
    context_k: int = 5,
    top_k: int = 10,
    verbose: bool = True,
    use_double_prompt: bool = False,
) -> str:
    model_adaptor_name = model_adaptor.name()
    rag_logger.info(f"ask() using '{model_adaptor_name}': {question[:80]}...")

    q, truncated = truncate_question(question)

    t0 = time.time()
    refs = await search_references(state, q, top_k=top_k, verbose=verbose)
    docs = refs.get("results", [])
    context = build_context_improved(docs, q, context_k=context_k)
    rag_logger.info(f"chat search_references results: {docs}")

    if verbose:
        rag_logger.info(f"Retrieved context in {time.time() - t0:.2f}s")

    if use_double_prompt:
        prompt = (
            f"{model_adapters.SYSTEM_PROMPT}\n\n"
            "Agent Question:\n"
            f"{q}\n"
            f"{q}\n\n"
            "DOCUMENTS (internal — do not mention they exist):\n"
            f"{context}"
        )
    else:
        prompt = (
            f"{model_adapters.SYSTEM_PROMPT}\n\n"
            "Agent Question:\n"
            f"{q}\n\n"
            "DOCUMENTS (internal — do not mention they exist):\n"
            f"{context}"
        )

    rag_logger.info(f"-- PROMPT --\n{prompt} \n-- END PROMPT --")

    answer = model_adaptor.generate(prompt, temperature=0.3, max_new_tokens=768)

    if truncated:
        answer = "(Question truncated)\n\n" + answer

    return answer.strip()


async def ask_model_only(
    model_adaptor,
    state: RetrievalState,
    question: str,
    *,
    verbose: bool = True,
    use_double_prompt: bool = False,
) -> str:
    model_adaptor_name = model_adaptor.name()
    rag_logger.info(f"ask_model_only() using '{model_adaptor_name}': {question[:80]}...")

    q, truncated = truncate_question(question)

    if use_double_prompt:
        prompt = (
            f"{model_adapters.SYSTEM_PROMPT}\n\n"
            "Agent Question:\n"
            f"{q}\n"
            f"{q}\n"
        )
    else:
        prompt = (
            f"{model_adapters.SYSTEM_PROMPT}\n\n"
            "Agent Question:\n"
            f"{q}\n"
        )

    rag_logger.info(f"-- PROMPT --\n{prompt} \n-- END PROMPT --")

    if verbose:
        rag_logger.info(f"-- PROMPT (model-only) --\n{prompt}\n-- END PROMPT --")

    answer = model_adaptor.generate(prompt, temperature=0.3, max_new_tokens=768)

    if truncated:
        answer = "(Question truncated)\n\n" + answer

    return str(answer or "").strip()


# ------------------ QUEUEING ------------------

job_queue: List[QueuedJob] = []
job_lock = threading.Lock()

outgoing_queue: List[QueuedResponse] = []
outgoing_lock = threading.Lock()

inflight_users = {}
inflight_lock = threading.Lock()


def queue_job(user_id, job_id, msg, prompt) -> bool:
    queued_job = QueuedJob(user_id, job_id, msg, prompt)
    with job_lock:
        job_queue.append(queued_job)
        return len(job_queue)


def fetch_queued_job_info(user_id) -> Tuple[int, Optional[QueuedJob]]:
    with job_lock:
        for i, item in enumerate(job_queue):
            if user_id == item.user_id:
                return i, item
    return 0, None


def has_queued_job() -> bool:
    with job_lock:
        return len(job_queue) > 0


def get_next_queued_job() -> Optional[QueuedJob]:
    with job_lock:
        if len(job_queue) == 0:
            return None
        return job_queue[0]


def pop_next_queued_job():
    with job_lock:
        if len(job_queue) == 0:
            return
        job_queue.pop(0)


def job_queue_len():
    with job_lock:
        return len(job_queue)


def queue_outgoing(response) -> bool:
    if not isinstance(response, QueuedResponse):
        raise RuntimeError("queue_outgoing requires QueuedResponse param")

    with outgoing_lock:
        outgoing_queue.append(response)
        qsize = len(outgoing_queue)

    rag_logger.info(
        f"Outgoing resp with job_id {response.job_id} was queued successfully for sending. Curr depth: {qsize}"
    )
    return qsize


def fetch_queued_outgoing_info(user_id) -> Optional[QueuedResponse]:
    with outgoing_lock:
        for i, item in enumerate(outgoing_queue):
            if user_id == item.user_id:
                del outgoing_queue[i]
                rag_logger.info(
                    f"Fetched queue outgoing info for job_id {item.job_id}, user_id {item.user_id}"
                )
                return item
    return None


def outgoing_queue_len():
    with outgoing_lock:
        return len(outgoing_queue)



# ------------------ MAIN ------------------

async def main():
    state = boot()
    refs = await search_references(state, "death squads in Haiti", top_k=10)
    rag_logger.info(json.dumps(refs, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    asyncio.run(main())



