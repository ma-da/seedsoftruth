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
import model_prompts
import rag_cleaner
import utils

rag_logger = logging_config.get_logger("rag")

TRINEDAY_TOKEN = "Trine Day Publishing"
COPYRIGHT_BLOCK_MSG = "This text protected by copyright."

# ------------------ MODEL ADAPTERS ------------------
hf_llm_model = model_adapters.LLMFactory.create("hf")
deep_infra_llm_model = model_adapters.LLMFactory.create("deepinfra")
spark_llm_model = model_adapters.LLMFactory.create("spark")
sim_model = model_adapters.LLMFactory.create("sim")

DEFAULT_MODEL_TYPE = os.getenv("MODEL_ADAPTER", "hf").strip().lower()

if DEFAULT_MODEL_TYPE == "spark":
    llm_model = spark_llm_model
elif DEFAULT_MODEL_TYPE == "deepinfra":
    llm_model = deep_infra_llm_model
else:
    llm_model = hf_llm_model

llm_models = [hf_llm_model, deep_infra_llm_model, spark_llm_model]


def get_model_type(type: str) -> model_adapters.LLMStrategy:
    global llm_model, hf_llm_model, deep_infra_llm_model, spark_llm_model, sim_model

    if type == "default":
        return llm_model

    if not model_adapters.is_valid_model_type(type):
        raise ValueError(f"Invalid model type: {type}")

    if type == "hf":
        return hf_llm_model
    elif type == "deepinfra":
        return deep_infra_llm_model
    elif type == "spark":
        return spark_llm_model
    elif type == "sim":
        return sim_model
    else:
        raise ValueError(f"Unknown model type: {type}")
        
rag_logger.info(f"DEFAULT_MODEL_TYPE={DEFAULT_MODEL_TYPE}")
rag_logger.info(f"llm_model selected: {llm_model.name()}")
rag_logger.info(f"available model names: {[m.name() for m in llm_models]}")

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
HYBRID_DB_PATH = Path(os.getenv("HYBRID_DB_PATH", "./data/gamma_master_hybrid_fts_stage2.db"))
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
    subsets: List[str] | None = None
    rag_algo_choice: int = 0


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
    subsets: List[str] | None = None


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


def extract_canonical_entity_terms(query: str, state: RetrievalState) -> List[str]:
    """Flat-list view of extract_canonical_entity_terms_typed (backward-compat)."""
    return [term for term, _category in extract_canonical_entity_terms_typed(query, state)]


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


# Function words and interrogatives that should not contribute to query/doc
# token-overlap counts. Used by build_context_v2 to avoid the "one-token
# match on a polysemous word" failure mode (e.g., a query like "Who killed
# JFK?" matching JFK-Airport sentences via the lone token "jfk").
_QUERY_STOPWORDS: frozenset = frozenset({
    "who", "what", "when", "where", "why", "how", "which", "whom",
    "is", "are", "was", "were", "be", "being", "been", "am",
    "do", "does", "did", "doing", "done",
    "have", "has", "had", "having",
    "can", "could", "should", "would", "will", "may", "might", "must", "shall",
    "the", "and", "but", "not", "nor",
    "of", "for", "with", "into", "from", "about", "between", "among",
    "this", "that", "these", "those", "such",
    "they", "them", "their", "theirs",
    "you", "your", "yours", "our", "ours",
    "tell", "show", "explain", "describe", "give", "list",
    "any", "all", "each", "every", "both", "either", "neither", "some",
    "than", "then", "very", "much", "more", "most", "less",
})


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

def _fts_positive_score_simple(bm25_score: float) -> float:
    # FTS5 bm25 has negative scores as better, just invert it
    return -float(bm25_score)

def _sqlite_row_to_result(
    row: sqlite3.Row,
    hybrid_score: float,
    entity_score: float,
    fulltext_score: float,
    raw_score: float,
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
    subsets: List[str] = None,
) -> List[Dict[str, Any]]:
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
        #score = _fts_positive_score(row["bm25_score"])
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
        #score = _fts_positive_score(row["bm25_score"])
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

    for lookup_id, hybrid_score, entity_score, fulltext_score, raw_score in ranked:
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
        title_penalty = _generic_title_penalty(title)

        adjusted_hybrid_score = (
            hybrid_score
            - title_penalty
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



# ------------------ HYBRID SQLITE SEARCH 2 ------------------

import math
from typing import List, Dict, Any


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



#
# ----------------------------
# Helpers
# ----------------------------
#

def _build_phrase_query(
    raw_query: str
) -> str:
    """
    Basic exact phrase query.
    Adjust to your tokenizer rules.
    """
    q = str(raw_query or "").strip()

    if len(q.split()) < 4:
        return ""

    return f'"{q}"'


def _get_cached_chunk_count(
    state,
    cur
):
    if getattr(
        state,
        "_cached_chunk_count",
        None
    ):
        return state._cached_chunk_count

    cur.execute(
        """
        SELECT COUNT(*) AS n
        FROM chunks
        """
    )

    n = int(
        cur.fetchone()["n"]
    )

    state._cached_chunk_count = n

    return n



def _load_entity_df(
    cur,
    query_entities,
):
    if not query_entities:
        return {}

    ph=",".join(
        "?"*len(query_entities)
    )

    rows=list(
        cur.execute(
            f"""
            SELECT
                e.canonical_name,
                COUNT(
                  DISTINCT ce.chunk_lookup_id
                ) AS df
            FROM entities e
            JOIN chunk_entities ce
              ON e.entity_id=ce.entity_id
            WHERE e.canonical_name IN ({ph})
            GROUP BY e.entity_id
            """,
            query_entities,
        )
    )

    return {
        r["canonical_name"]: int(
            r["df"]
        )
        for r in rows
    }



def _candidate_entities(
    cur,
    ids
):
    if not ids:
        return {}

    ph=",".join(
      "?"*len(ids)
    )

    rows=list(
        cur.execute(
            f"""
            SELECT
                ce.chunk_lookup_id,
                e.canonical_name,
                e.type
            FROM chunk_entities ce
            JOIN entities e
             ON ce.entity_id=e.entity_id
            WHERE ce.chunk_lookup_id
              IN ({ph})
            """,
            ids
        )
    )

    out={}

    for r in rows:

        cid=int(
           r["chunk_lookup_id"]
        )

        out.setdefault(
           cid,
           []
        ).append(
           (
             r["canonical_name"],
             r["type"],
           )
        )

    return out



def _entity_structured_score(
    query_entities,
    chunk_entities,
    entity_df,
    corpus_chunks,
):
    if not query_entities:
        return (0.0,0.0,0.0)

    chunk_names={
       e[0]
       for e in chunk_entities
    }

    matched=[
      e for e in query_entities
      if e in chunk_names
    ]

    if not matched:
        return (0.0,0.0,0.0)


    #
    # softer than squared
    #
    coverage=(
       len(matched)
       / len(query_entities)
    )

    coverage_score=(
       coverage ** 1.5
    )


    #
    # capped rarity
    #
    rarity=0.0

    type_map=dict(
      chunk_entities
    )

    for e in matched:

        df=max(
          entity_df.get(e,1),
          1
        )

        idf=min(
          math.log(
            corpus_chunks/df
          ),
          IDF_CAP
        )

        rarity += (
            ENTITY_TYPE_WEIGHT.get(
                type_map.get(
                    e
                ),
                .5
            )
            * idf
        )


    rarity/=max(
       len(matched),
       1
    )


    #
    # smaller cooccur
    #
    cooccur_bonus=(
       COOCCUR_BONUS
       if len(matched)>=2
       else 0.0
    )

    return (
       coverage_score,
       rarity,
       cooccur_bonus
    )


def _hybrid_search_sqlite2(
        state,
        *,
        entity_terms,
        fulltext_query,
        top_k,
        require_all_entities=True,
        subsets=None,
)->List[Dict[str,Any]]:

    conn=_get_sqlite_conn(
        state
    )

    cur=conn.cursor()


    entity_query=build_entity_match_query(
        entity_terms,
        require_all=require_all_entities
    )


    raw_ft_query=str(
       fulltext_query or ""
    ).strip()


    phrase_fulltext_query=(
      _build_phrase_query(
        raw_ft_query
      )
    )

    strict_fulltext_query=(
      build_fulltext_query(
         raw_ft_query,
         require_all=True
      )
    )

    broad_fulltext_query=(
      build_fulltext_query(
         raw_ft_query,
         require_all=False
      )
    )


    #
    # ----------------------
    # Candidate generation
    # ----------------------
    #

    entity_rows=[]

    if entity_query:

        entity_rows=list(
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
              (
                 entity_query,
                 HYBRID_ENTITY_LIMIT
              ),
            )
        )


    fulltext_rows=[]
    fulltext_query_used=""


    #
    # phrase first
    #
    if phrase_fulltext_query:

        fulltext_rows=list(
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
             (
               phrase_fulltext_query,
               HYBRID_FULLTEXT_LIMIT
             ),
           )
        )

        if fulltext_rows:
            fulltext_query_used=(
               phrase_fulltext_query
            )


    if (
      not fulltext_rows
      and strict_fulltext_query
    ):

        fulltext_rows=list(
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
              (
                strict_fulltext_query,
                HYBRID_FULLTEXT_LIMIT
              ),
            )
        )

        if fulltext_rows:
            fulltext_query_used=(
               strict_fulltext_query
            )


    if (
      not fulltext_rows
      and broad_fulltext_query
      and broad_fulltext_query!=strict_fulltext_query
    ):
        fulltext_rows=list(
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
                (
                  broad_fulltext_query,
                  HYBRID_FULLTEXT_LIMIT
                ),
            )
        )

        if fulltext_rows:
            fulltext_query_used=(
              broad_fulltext_query
            )


    #
    # ----------------------
    # Merge hybrid
    # ----------------------
    #

    merged={}

    for rows,field in (
       (entity_rows,"entity_score"),
       (fulltext_rows,"fulltext_score")
    ):

        for row in rows:

            lookup_id=int(
               row["lookup_id"]
            )

            score=(
              _fts_positive_score_simple(
                 row["bm25_score"]
              )
            )

            merged.setdefault(
              lookup_id,
              {
                "entity_score":0.0,
                "fulltext_score":0.0,
                "raw_score":0.0,
              }
            )

            merged[lookup_id][field]=max(
              merged[lookup_id][field],
              score
            )

            merged[lookup_id]["raw_score"]=max(
               merged[lookup_id]["raw_score"],
               score
            )


    ranked=[]

    for lookup_id,parts in merged.items():

        hybrid_score=(
            HYBRID_ENTITY_WEIGHT_V2
            * parts["entity_score"]

            + HYBRID_FULLTEXT_WEIGHT
            * parts["fulltext_score"]
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


    ranked.sort(
      key=lambda x:x[1],
      reverse=True
    )


    prefilter_limit=max(
       top_k*20,
       200
    )

    prefilter_limit=min(
       prefilter_limit,
       2000
    )

    ranked=ranked[:prefilter_limit]

    if not ranked:
        return []


    ids=[
      r[0]
      for r in ranked
    ]

    ph=",".join(
      "?" for _ in ids
    )

    query=f"""
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

    params=list(ids)

    if subsets:
        s_ph=",".join(
          "?" for _ in subsets
        )

        query+=(
         f" AND subset_name IN ({s_ph})"
        )

        params.extend(
          subsets
        )


    meta_rows=list(
      cur.execute(
         query,
         params
      )
    )

    meta_map={
      int(r["lookup_id"]):r
      for r in meta_rows
    }


    #
    # structured metadata
    #
    entity_df=_load_entity_df(
        cur,
        entity_terms
    )

    candidate_entity_map=(
      _candidate_entities(
         cur,
         ids
      )
    )

    corpus_chunks=(
      _get_cached_chunk_count(
         state,
         cur
      )
    )


    #
    # ----------------------
    # Rescore
    # ----------------------
    #

    rescored=[]

    for (
       lookup_id,
       hybrid_score,
       entity_score,
       fulltext_score,
       raw_score
    ) in ranked:

        row=meta_map.get(
          lookup_id
        )

        if row is None:
            continue


        title=str(
          row["title"] or ""
        )

        fulltext_text=str(
          row["fulltext_text"] or ""
        )


        if _looks_too_punct_noisy(
            title,
            fulltext_text
        ):
            continue


        keyword_bonus=(
           _keyword_overlap_bonus(
             raw_ft_query,
             title,
             fulltext_text
           )
        )

        phrase_bonus=(
           _exact_phrase_bonus(
              raw_ft_query,
              title,
              fulltext_text
           )
        )

        title_penalty=(
          _generic_title_penalty(
             title
          )
        )


        #
        # anchor penalty
        #
        anchor_penalty=0.0

        if entity_terms:
            if not _has_early_anchor(
                entity_terms,
                title,
                fulltext_text
            ):
                anchor_penalty=(
                  NO_ANCHOR_PENALTY
                )


        chunk_entities=(
           candidate_entity_map.get(
             lookup_id,
             []
           )
        )

        (
          coverage_score,
          rarity_score,
          cooccur_bonus
        )=_entity_structured_score_simple(
            entity_terms,
            chunk_entities,
            entity_df,
            corpus_chunks
        )


        #
        # soft proximity
        #
        proximity_bonus=0.0

        if len(entity_terms)>=2:

            if _has_entity_proximity_match(
                entity_terms,
                fulltext_text,
                max_chars=250
            ):
                proximity_bonus=(
                  PROX_BONUS_NEAR
                )

            elif _has_entity_proximity_match(
                entity_terms,
                fulltext_text,
                max_chars=700
            ):
                proximity_bonus=(
                  PROX_BONUS_FAR
                )


        adjusted_score=(
            hybrid_score

            + ENTITY_COVERAGE_WEIGHT
               * coverage_score

            + ENTITY_RARITY_WEIGHT
               * rarity_score

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


    rescored.sort(
      key=lambda x:x[1],
      reverse=True
    )

    rescored=rescored[:top_k]


    results=[]

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
      proximity_bonus
    ) in rescored:

        row=meta_map[
          lookup_id
        ]

        result=_sqlite_row_to_result(
            row,
            adjusted_score,
            entity_score,
            fulltext_score,
            raw_score
        )


        #
        # Debug
        #
        result["coverage_score"]=float(
            coverage_score
        )

        result["rarity_score"]=float(
            rarity_score
        )

        result["cooccur_bonus"]=float(
            cooccur_bonus
        )

        result["proximity_bonus"]=float(
            proximity_bonus
        )

        result["keyword_bonus"]=float(
            keyword_bonus
        )

        result["phrase_bonus"]=float(
            phrase_bonus
        )

        result["hybrid_score"]=float(
            adjusted_score
        )

        results.append(
           result
        )


    return results

# ------------------ HYBRID SEARCH SQLITE V3 ------------------

import math
import re
from typing import List, Dict, Any, Set, Tuple, Optional


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
    "kill": {"kill", "killed", "killing", "murder", "murdered", "assassinate", "assassinated", "slay", "slain", "execute", "executed", "shoot", "shot"},
    "murder": {"murder", "murdered", "kill", "killed", "assassinate", "assassinated", "shot"},
    "assassinate": {"assassinate", "assassinated", "murder", "murdered", "kill", "killed", "shot"},
    "shoot": {"shoot", "shot", "gunman", "gunmen", "fire", "fired", "wound", "wounded"},
    "poison": {"poison", "poisoned", "toxin", "toxic", "overdose"},
    "attack": {"attack", "attacked", "assault", "assaulted", "strike", "struck"},
    "harm": {"harm", "harmed", "injure", "injured", "damage", "damaged"},
    "die": {"die", "died", "death", "dead", "killed", "fatal"},

    # agency / responsibility / causation
    "cause": {"cause", "caused", "causing", "lead", "led", "trigger", "triggered", "produce", "produced", "result", "resulted"},
    "trigger": {"trigger", "triggered", "cause", "caused", "spark", "sparked"},
    "lead": {"lead", "led", "cause", "caused", "resulted"},
    "blame": {"blame", "blamed", "responsible", "culpable", "fault"},
    "responsible": {"responsible", "culpable", "accountable", "blame", "blamed"},

    # funding / money / support
    "fund": {"fund", "funded", "funding", "finance", "financed", "financing", "bankroll", "bankrolled", "sponsor", "sponsored"},
    "finance": {"finance", "financed", "fund", "funded", "bankroll", "bankrolled"},
    "pay": {"pay", "paid", "payment", "payments", "finance", "funded"},
    "sponsor": {"sponsor", "sponsored", "funded", "backed", "supported"},
    "back": {"back", "backed", "support", "supported", "sponsor", "sponsored"},

    # command / authorization / planning
    "order": {"order", "ordered", "command", "commanded", "authorize", "authorized", "direct", "directed", "instruct", "instructed"},
    "authorize": {"authorize", "authorized", "approve", "approved", "sanction", "sanctioned"},
    "direct": {"direct", "directed", "order", "ordered", "command", "commanded"},
    "plan": {"plan", "planned", "plot", "plotted", "scheme", "schemed", "conspire", "conspired"},
    "organize": {"organize", "organized", "coordinate", "coordinated", "orchestrate", "orchestrated"},
    "orchestrate": {"orchestrate", "orchestrated", "coordinate", "coordinated", "organize", "organized"},

    # control / influence / manipulation
    "control": {"control", "controlled", "influence", "influenced", "manipulate", "manipulated", "steer", "steered"},
    "influence": {"influence", "influenced", "pressure", "pressured", "lobby", "lobbied", "shape", "shaped"},
    "manipulate": {"manipulate", "manipulated", "rig", "rigged", "distort", "distorted"},
    "coerce": {"coerce", "coerced", "force", "forced", "pressure", "pressured"},

    # creation / origin / invention / founding
    "create": {"create", "created", "found", "founded", "establish", "established", "form", "formed", "invent", "invented"},
    "found": {"found", "founded", "create", "created", "establish", "established"},
    "establish": {"establish", "established", "create", "created", "found", "founded"},
    "invent": {"invent", "invented", "develop", "developed", "create", "created"},
    "develop": {"develop", "developed", "build", "built", "create", "created"},

    # discovery / revelation / investigation
    "discover": {"discover", "discovered", "find", "found", "uncover", "uncovered", "reveal", "revealed"},
    "reveal": {"reveal", "revealed", "disclose", "disclosed", "expose", "exposed", "uncover", "uncovered"},
    "investigate": {"investigate", "investigated", "probe", "probed", "inquiry", "inquiries"},
    "expose": {"expose", "exposed", "reveal", "revealed", "uncover", "uncovered"},

    # accusation / claim / testimony
    "accuse": {"accuse", "accused", "allege", "alleged", "claim", "claimed", "charge", "charged"},
    "allege": {"allege", "alleged", "accuse", "accused", "claim", "claimed"},
    "claim": {"claim", "claimed", "allege", "alleged", "assert", "asserted"},
    "testify": {"testify", "testified", "testimony", "witness", "witnessed"},
    "admit": {"admit", "admitted", "confess", "confessed", "acknowledge", "acknowledged"},

    # concealment / coverup / suppression
    "hide": {"hide", "hid", "hidden", "conceal", "concealed", "cover", "covered", "suppress", "suppressed"},
    "conceal": {"conceal", "concealed", "hide", "hidden", "coverup", "cover-up"},
    "suppress": {"suppress", "suppressed", "censor", "censored", "bury", "buried"},
    "censor": {"censor", "censored", "suppress", "suppressed", "ban", "banned"},
    "cover": {"cover", "covered", "coverup", "cover-up", "conceal", "concealed"},

    # relationship / connection
    "connect": {"connect", "connected", "link", "linked", "associate", "associated", "relate", "related"},
    "link": {"link", "linked", "connect", "connected", "tie", "tied", "associate", "associated"},
    "associate": {"associate", "associated", "link", "linked", "connect", "connected"},
    "meet": {"meet", "met", "meeting", "encounter", "encountered"},
    "work": {"work", "worked", "collaborate", "collaborated", "cooperate", "cooperated"},

    # membership / affiliation
    "join": {"join", "joined", "member", "membership", "belong", "belonged"},
    "belong": {"belong", "belonged", "member", "membership", "affiliated"},
    "affiliate": {"affiliate", "affiliated", "associate", "associated", "member"},

    # legal / official actions
    "arrest": {"arrest", "arrested", "detain", "detained", "custody"},
    "charge": {"charge", "charged", "indict", "indicted", "accuse", "accused"},
    "convict": {"convict", "convicted", "sentence", "sentenced", "guilty"},
    "sue": {"sue", "sued", "lawsuit", "litigation"},
    "ban": {"ban", "banned", "prohibit", "prohibited", "outlaw", "outlawed"},

    # transmission / spread / dissemination
    "spread": {"spread", "spreading", "transmit", "transmitted", "disseminate", "disseminated", "circulate", "circulated"},
    "publish": {"publish", "published", "release", "released", "distribute", "distributed"},
    "leak": {"leak", "leaked", "disclose", "disclosed", "release", "released"},

    # comparison / identity / attribution
    "be": {"is", "was", "were", "are", "be", "being", "become", "became"},
    "identify": {"identify", "identified", "name", "named", "recognize", "recognized"},
    "name": {"name", "named", "identify", "identified"},
}


RELATION_VERBS = set(RELATION_LEMMA_EXPANSIONS.keys())


# ============================================================
# Helpers
# ============================================================

def _build_phrase_query(raw_query: str) -> str:
    q = str(raw_query or "").strip()
    if len(q.split()) < 4:
        return ""
    # Escape quotes for FTS phrase use.
    q = q.replace('"', '""')
    return f'"{q}"'


def _safe_lower_text(text: str) -> str:
    return str(text or "").lower()


def _contains_any_term(text_lc: str, terms: Set[str]) -> bool:
    for term in terms:
        if not term:
            continue
        # Conservative word-ish boundary for normal words; substring fallback for hyphenated forms.
        if re.search(r"\b" + re.escape(term.lower()) + r"\b", text_lc):
            return True
    return False


def _get_cached_chunk_count(state, cur) -> int:
    cached = getattr(state, "_cached_chunk_count", None)
    if cached:
        return int(cached)

    cur.execute("SELECT COUNT(*) AS n FROM chunks")
    n = int(cur.fetchone()["n"])
    state._cached_chunk_count = n
    return n


def _load_entity_df(cur, query_entities: List[str]) -> Dict[str, int]:
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


def _candidate_entities(cur, ids: List[int]) -> Dict[int, List[Tuple[str, str]]]:
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
    """
    LEGACY / SIMPLE VERSION (original 3-return)
    Used in early v2. No matched_entity_count.
    """
    if not query_entities:
        return 0.0, 0.0, 0.0

    chunk_names = {e[0] for e in chunk_entities}
    matched = [e for e in query_entities if e in chunk_names]

    if not matched:
        return 0.0, 0.0, 0.0

    coverage = len(matched) / len(query_entities)
    coverage_score = coverage ** 1.5

    rarity = 0.0
    type_map = dict(chunk_entities)
    safe_corpus_chunks = max(int(corpus_chunks or 1), 1)

    for e in matched:
        df = max(entity_df.get(e, 1), 1)
        idf = math.log(safe_corpus_chunks / df) if safe_corpus_chunks > df else 0.0
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
    """
    ENHANCED VERSION (current 4-return)
    Returns matched_entity_count for debugging, null-evidence detection, etc.
    Recommended for new experiments.
    """
    if not query_entities:
        return 0.0, 0.0, 0.0, 0

    chunk_names = {e[0] for e in chunk_entities}
    matched = [e for e in query_entities if e in chunk_names]

    if not matched:
        return 0.0, 0.0, 0.0, 0

    coverage = len(matched) / len(query_entities)
    coverage_score = coverage ** 1.5

    rarity = 0.0
    type_map = dict(chunk_entities)
    safe_corpus_chunks = max(int(corpus_chunks or 1), 1)

    for e in matched:
        df = max(entity_df.get(e, 1), 1)
        idf = math.log(safe_corpus_chunks / df) if safe_corpus_chunks > df else 0.0
        idf = min(idf, IDF_CAP)
        rarity += ENTITY_TYPE_WEIGHT.get(type_map.get(e), 0.5) * idf

    rarity /= max(len(matched), 1)
    cooccur_bonus = COOCCUR_BONUS if len(matched) >= 2 else 0.0

    return coverage_score, rarity, cooccur_bonus, len(matched)


def _analyze_query_spacy(nlp, raw_query: str) -> Dict[str, Any]:
    """
    Uses a globally available spaCy model named `nlp`.

    Returns a soft query-structure profile:
      - predicate lemma
      - predicate variants
      - content terms
      - relation mode
      - interrogative type
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
            if tok.pos_ == "VERB" and tok.lemma_.lower() not in {"be", "do", "have"}:
                predicate = tok.lemma_.lower()
                break

    wh_terms = {tok.lower_ for tok in doc if tok.tag_ in {"WP", "WDT", "WRB"} or tok.lower_ in {"who", "what", "when", "where", "why", "how", "which"}}

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
        predicate_variants = RELATION_LEMMA_EXPANSIONS.get(predicate, {predicate})

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
    """
    Soft score for whether the chunk actually supports the predicate/relation.
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

    for lookup_id, hybrid_score, entity_score, fulltext_score, raw_score in ranked:
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
            if coverage_score >= 0.80 and predicate_support < RELATION_MODE_MIN_PRED_SUPPORT:
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
            r.get("predicate_support", 0.0)
            for r in top_n
        ) / max(len(top_n), 1)

        mean_coverage = sum(
            r.get("coverage_score", 0.0)
            for r in top_n
        ) / max(len(top_n), 1)

        if mean_coverage >= 0.70 and mean_predicate_support < RELATION_MODE_MIN_PRED_SUPPORT:
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
    """
    Two-condition min-gate.

    Returns (passed, reason) where reason is one of:
        "pass"
        "no_results"                    — retrieval returned nothing
        "score_below_floor"             — top-1 BM25 < floor (catches cadmium/schooner/swallow probes
                                          and very low-confidence in-domain queries)
        "no_entities_broad_or_only"     — no canonical entities matched AND the FTS branch fell
                                          through to broad-OR (catches the Switzerland-style probes
                                          where score is high but match is incidental)
    """
    if top1_score is None or (isinstance(top1_score, float) and math.isnan(top1_score)):
        return False, "no_results"
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
    """
    v4 = v2's retrieval + a query-level min-gate.

    If the gate passes, returns v2's ranked results unchanged. If the gate
    declines, returns []. Optional `meta_out` dict (mutated in place) is
    populated with gate decision metadata so callers can surface it in
    API responses or logs.

    Parameters
    ----------
    entity_terms
        All canonical entity terms — used for v2's underlying entity-FTS
        retrieval (locations included; we still want them to *match* docs).
    non_location_entity_terms
        Filtered list excluding location-typed entities. Used by the gate's
        condition-B check ("no entities + broad-OR only"). When `None` (e.g.,
        legacy callers), defaults to `entity_terms` and the gate behaves like
        the un-refined version.
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
    nlet = non_location_entity_terms if non_location_entity_terms is not None else entity_terms
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
            reason, top1_str, n_ent, n_non_loc, fts_branch, len(results),
        )
        return []

    rag_logger.info(
        "v4 min_gate PASS: top1=%.2f n_ent=%d n_non_loc=%d fts_branch=%s n=%d",
        top1, n_ent, n_non_loc, fts_branch, len(results),
    )
    return results


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
    rag_algo_choice: int = 0, # zero is default
    **_unused,
) -> Dict[str, Any]:
    q = (query or "").strip()
    if not q:
        return {"query": query, "num_results": 0, "results": [], "message": "Empty query."}

    entity_source = (entity_source_query or q).strip()
    fulltext_source = (fulltext_query or q).strip()

    typed_entity_terms = extract_canonical_entity_terms_typed(entity_source, state)
    entity_terms = [t for t, _ in typed_entity_terms]
    non_location_entity_terms = [t for t, cat in typed_entity_terms if cat != "locations"]
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


def build_context_v2(
    docs: List[Dict[str, Any]],
    query: str,
    *,
    state: Optional[RetrievalState] = None,
    context_k: int = 10,
    max_snips_per_doc: int = 5,
    max_sent_per_doc: int = 50,
    window_radius: int = 1,
    min_sentence_overlap: int = 2,
    min_doc_kept_sentences: int = 2,
    total_snippet_budget: int = 15,
) -> str:
    """
    Snippet extractor with global (cross-doc) snippet selection.

    Pipeline:

      Phase 1 — per-doc sentence scoring + doc-level filter
        Each candidate doc (up to `context_k`) is split into sentences. Each
        sentence is scored on filtered query-term overlap (with stopwords
        stripped) plus a bonus for canonical-entity matches (when `state`
        is provided). A sentence "qualifies" iff overlap >= min_sentence_overlap
        OR it contains a canonical entity. A doc is kept only if at least
        `min_doc_kept_sentences` of its sentences qualify — this filter
        catches polysemous-match noise (e.g., JFK-Airport docs for a JFK
        assassination query).

      Phase 2 — global snippet selection
        All qualifying sentences from all kept docs go into a single global
        pool, sorted by score. The top `total_snippet_budget` are picked,
        respecting a per-doc cap of `max_snips_per_doc` so one verbose doc
        can't dominate.

      Phase 3 — regroup, expand windows, dedup
        Selected sentence indices are regrouped by doc. Each is expanded
        into a window of `window_radius` neighboring sentences; overlapping
        windows in the same doc merge. Trigram-level dedup runs across
        all final blocks.

    Three behavior changes from build_context_improved this addresses:

      1. Stopword filter on query terms. "who", "what", "the", "is" and
         friends are stripped before token-overlap matching, so they don't
         pad relevance scores on unrelated content.

      2. Stricter retention via canonical-entity awareness. A single-token
         match on a polysemous word like "jfk" no longer suffices — the
         sentence needs either ≥2 query-term overlap OR a canonical-entity
         hit (which disambiguates JFK-the-person from JFK-the-airport).

      3. Global snippet selection. Instead of a fixed quota per doc, the
         best content surfaces from across all kept docs. A doc with one
         exceptional sentence outranks a doc with five mediocre ones.

    `state` is optional. When omitted, the canonical-entity path is skipped
    — sentences must clear the overlap threshold on their own. Output
    format is unchanged: a sequence of
    <doc id="..." url="..." score="..."><snippets>...</snippets></doc> blocks.
    """
    raw_terms = _to_term_set(query)
    terms = {t for t in raw_terms if t not in _QUERY_STOPWORDS}
    # If the query was entirely stopwords (rare), fall back to the unfiltered
    # set rather than retaining nothing.
    if not terms:
        terms = raw_terms

    # Canonical entity tokens from the query (lowercase, underscore-split).
    canonical_terms: set = set()
    if state is not None:
        try:
            for ent in extract_canonical_entity_terms(query, state):
                for token in ent.replace("_", " ").lower().split():
                    if len(token) > 2:
                        canonical_terms.add(token)
        except Exception as e:
            rag_logger.warning(f"build_context_v2: entity extraction failed: {e}")

    rag_logger.info(
        f"build_context_v2: filtered_terms={sorted(terms)} "
        f"canonical_terms={sorted(canonical_terms)} "
        f"context_k={context_k} budget={total_snippet_budget} "
        f"min_overlap={min_sentence_overlap} min_doc_keep={min_doc_kept_sentences}"
    )

    picked = (docs or [])[: int(context_k)]

    # ---- Phase 1: per-doc sentence scoring + doc-level filter ----
    # passing_docs[i] = {"doc": ..., "sents": [...], "qualifying": [(sent_idx, score), ...]}
    passing_docs: Dict[int, Dict[str, Any]] = {}

    for i, d in enumerate(picked):
        text = (d.get("text") or d.get("snippet") or "").strip()
        if not text:
            continue

        text = rag_cleaner.clean_text_for_rag(text)
        sents = _sentence_split(text)[: int(max_sent_per_doc)]
        if not sents:
            continue

        qualifying: List[Tuple[int, int]] = []
        canon_hits = 0
        for idx, s in enumerate(sents):
            sent_tokens = set(_tokenize_simple(s.lower()))
            overlap = len(terms & sent_tokens)
            has_canon = bool(canonical_terms & sent_tokens)
            if overlap >= min_sentence_overlap or has_canon:
                # Sort score: term overlap + bonus for canonical-entity match.
                rank_score = overlap + (2 if has_canon else 0)
                qualifying.append((idx, rank_score))
                if has_canon:
                    canon_hits += 1

        if len(qualifying) < int(min_doc_kept_sentences):
            doc_id_hint = (d.get("row_id") or d.get("title") or "?")[:24]
            rag_logger.info(
                f"build_context_v2: drop doc {i+1} ({doc_id_hint}) — "
                f"{len(qualifying)} sentences passed (need {min_doc_kept_sentences}, "
                f"canon_hits={canon_hits})"
            )
            continue

        passing_docs[i] = {"doc": d, "sents": sents, "qualifying": qualifying}

    if not passing_docs:
        rag_logger.info(f"build_context_v2: no docs survived filter (of {len(picked)} candidates)")
        return ""

    # ---- Phase 2: global snippet selection ----
    # Flatten all qualifying sentences from all passing docs, sort globally,
    # then pick top N respecting per-doc cap.
    global_pool: List[Tuple[int, int, int]] = []  # (doc_idx, sent_idx, score)
    for doc_idx, info in passing_docs.items():
        for sent_idx, score in info["qualifying"]:
            global_pool.append((doc_idx, sent_idx, score))
    global_pool.sort(key=lambda t: -t[2])  # highest score first

    selected_by_doc: Dict[int, List[int]] = {}  # doc_idx -> [sent_idx, ...]
    selected_count = 0
    for doc_idx, sent_idx, _score in global_pool:
        if selected_count >= int(total_snippet_budget):
            break
        cur = selected_by_doc.setdefault(doc_idx, [])
        if len(cur) >= int(max_snips_per_doc):
            continue
        cur.append(sent_idx)
        selected_count += 1

    rag_logger.info(
        f"build_context_v2: global pool size={len(global_pool)}, "
        f"selected={selected_count}/{total_snippet_budget} across "
        f"{len(selected_by_doc)} docs"
    )

    # ---- Phase 3: regroup, expand windows, dedup, emit ----
    global_tris: set = set()
    blocks: List[str] = []

    # Preserve the original doc order from `picked` so the prompt shows
    # higher-ranked retrieval results first.
    for doc_idx in sorted(selected_by_doc.keys()):
        info = passing_docs[doc_idx]
        d = info["doc"]
        sents = info["sents"]
        sent_indices = sorted(set(selected_by_doc[doc_idx]))

        # Expand each chosen sentence into a window of neighbors.
        windows: List[Tuple[int, int]] = []
        for idx in sent_indices:
            start = max(0, idx - window_radius)
            end = min(len(sents), idx + window_radius + 1)
            windows.append((start, end))
        windows.sort()

        # Merge overlapping/adjacent windows.
        merged: List[List[int]] = []
        for start, end in windows:
            if not merged:
                merged.append([start, end])
            else:
                if start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])

        # Trigram dedup across the full prompt's blocks (no near-duplicates).
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
        score_bm25 = d.get("score_bm25") or 0.0

        block = (
            f'<doc id="{doc_id}" url="{url}" score="{float(score_bm25):.2f}">\n'
            "<snippets>\n- " + "\n- ".join(kept_blocks) + "\n</snippets>\n"
            "</doc>"
        )
        blocks.append(block)

    rag_logger.info(
        f"build_context_v2: emitted {len(blocks)} doc blocks from "
        f"{len(selected_by_doc)} selected (of {len(passing_docs)} passing, "
        f"{len(picked)} candidates)"
    )
    return "\n".join(blocks)


# ------------------ HF ORCHESTRATION ------------------

def is_model_ready(timeout=model_adapters.MODEL_TIMEOUT_SECS) -> bool:
    return utils.do_async_to_sync(lambda: llm_model.is_model_ready(timeout=timeout))()


def is_model_type_ready(model_type: str, timeout=model_adapters.MODEL_TIMEOUT_SECS) -> bool:
    model = get_model_type(model_type)
    return utils.do_async_to_sync(lambda: model.is_model_ready(timeout=timeout))()


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
    context_k: int = 10,
    top_k: int = 10,
    verbose: bool = True,
    use_double_prompt: bool = False,
    subsets: List[str] = None,
    rag_algo_choice: int = 0,
    prompt_type: int = 0,
) -> str:
    model_adaptor_name = model_adaptor.name()
    rag_logger.info(f"ask() using '{model_adaptor_name}': {question[:80]}...")

    q, truncated = truncate_question(question)

    t0 = time.time()
    refs = await search_references(state, q, top_k=top_k, verbose=verbose, subsets=subsets, rag_algo_choice=rag_algo_choice)
    docs = refs.get("results", [])
    # build_context_v2: stricter filter + canonical-entity awareness — see
    # the "Who killed JFK?" -> JFK-Airport snippets failure that motivated
    # this change. Pass `state` so the entity disambiguation path is active.
    context = build_context_v2(docs, q, state=state, context_k=context_k)
    rag_logger.info(f"chat search_references results: {docs}")

    system_prompt = model_adapters.get_system_prompt(prompt_type)
    rag_logger.info(f"system prompt type: {prompt_type}")

    # Self-restatement variant: ask the model to reframe before answering. This
    # is a real prompting technique (improves coherence on ambiguous questions)
    # and replaces the prior "double prompt" mode that just typed the question
    # twice verbatim with no upside.
    if use_double_prompt:
        question_block = (
            "Agent Question:\n"
            f"{q}\n\n"
            "Before answering: restate the question in your own words and "
            "identify what would constitute strong evidence. Then answer.\n\n"
        )
    else:
        question_block = (
            "Agent Question:\n"
            f"{q}\n\n"
        )

    # DOCUMENTS section is included only when retrieval produced something.
    # When `context` is empty (e.g., v4's min-gate declined), no preamble is
    # added at all — the model just sees the system prompt + question and
    # is free to answer from general knowledge per the system prompt's rules.
    if str(context).strip():
        docs_block = (
            "Documents (use if helpful, otherwise ignore):\n"
            f"{context}"
        )
    else:
        docs_block = ""

    prompt = f"{system_prompt}\n\n{question_block}{docs_block}"

    prompt_tokens_len = None
    if verbose:
        prompt_tokens_len = utils.estimate_tokens(prompt)
        rag_logger.info(f"Retrieved context in {time.time() - t0:.2f}s")

    if verbose:
        context_tokens_len = utils.estimate_tokens(context)
        total_tokens_len = prompt_tokens_len + context_tokens_len

        rag_logger.info(f"prompt_total_toks {total_tokens_len}, prompt_toks {prompt_tokens_len}, context_toks {context_tokens_len}")
        rag_logger.info(f"-- PROMPT --\n{prompt} \n-- END PROMPT --")

    rag_logger.info(f"-- PROMPT --\n{prompt} \n-- END PROMPT --")

    answer = model_adaptor.generate(prompt,
                                    temperature=model_adapters.DEFAULT_TEMPERATURE,
                                    max_new_tokens=model_adapters.DEFAULT_MAX_TOKENS,
                                    )

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
    prompt_type: int = 0,
) -> str:
    model_adaptor_name = model_adaptor.name()
    rag_logger.info(f"ask_model_only() using '{model_adaptor_name}': {question[:80]}...")

    q, truncated = truncate_question(question)

    system_prompt = model_adapters.get_system_prompt(prompt_type)
    rag_logger.info(f"system prompt type: {prompt_type}")

    if use_double_prompt:
        prompt = (
            f"{system_prompt}\n\n"
            "Agent Question:\n"
            f"{q}\n"
            f"{q}\n"
        )
    else:
        prompt = (
            f"{system_prompt}\n\n"
            "Agent Question:\n"
            f"{q}\n"
        )

    rag_logger.info(f"-- PROMPT --\n{prompt} \n-- END PROMPT --")

    if verbose:
        rag_logger.info(f"-- PROMPT (model-only) --\n{prompt}\n-- END PROMPT --")

    answer = model_adaptor.generate(prompt,
                                    temperature=model_adapters.DEFAULT_TEMPERATURE,
                                    max_new_tokens=model_adapters.DEFAULT_MAX_TOKENS,
                                    )

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


def queue_job(
    user_id,
    job_id,
    msg,
    prompt,
    subsets: List[str],
    rag_algo_choice: int = 0,
) -> int:
    queued_job = QueuedJob(user_id, job_id, msg, prompt, subsets, rag_algo_choice)
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


