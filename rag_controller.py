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

    return {
        "row_id": row["chunk_id"] or row["lookup_id"],
        "source_url": src,
        "source": src,
        "title": row["title"] or "",
        "publisher": "",
        "found_on": "",
        "text": txt,
        "snippet": txt[:1200],
        "snippet_html": _snippet_html_from_text(txt, max_words=600),
        "score_bm25": float(hybrid_score),
        "entity_score": float(entity_score),
        "fulltext_score": float(fulltext_score),
    }


# ------------------ HYBRID SQLITE SEARCH ------------------

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

    entity_query = build_entity_match_query(entity_terms, require_all=require_all_entities)

    strict_fulltext_query = build_fulltext_query(fulltext_query, require_all=True)
    broad_fulltext_query = build_fulltext_query(fulltext_query, require_all=False)

    entity_rows = []
    if entity_query:
        entity_rows = list(cur.execute("""
            SELECT rowid AS lookup_id, bm25(entities_fts) AS bm25_score
            FROM entities_fts
            WHERE entities_fts MATCH ?
            ORDER BY bm25_score
            LIMIT ?
        """, (entity_query, HYBRID_ENTITY_LIMIT)))

    fulltext_rows = []
    fulltext_query_used = ""

    if strict_fulltext_query:
        fulltext_rows = list(cur.execute("""
            SELECT rowid AS lookup_id, bm25(fulltext_fts) AS bm25_score
            FROM fulltext_fts
            WHERE fulltext_fts MATCH ?
            ORDER BY bm25_score
            LIMIT ?
        """, (strict_fulltext_query, HYBRID_FULLTEXT_LIMIT)))
        fulltext_query_used = strict_fulltext_query

    if not fulltext_rows and broad_fulltext_query and broad_fulltext_query != strict_fulltext_query:
        fulltext_rows = list(cur.execute("""
            SELECT rowid AS lookup_id, bm25(fulltext_fts) AS bm25_score
            FROM fulltext_fts
            WHERE fulltext_fts MATCH ?
            ORDER BY bm25_score
            LIMIT ?
        """, (broad_fulltext_query, HYBRID_FULLTEXT_LIMIT)))
        fulltext_query_used = broad_fulltext_query

    merged: Dict[int, Dict[str, float]] = {}

    for row in entity_rows:
        lookup_id = int(row["lookup_id"])
        score = _fts_positive_score(row["bm25_score"])
        merged.setdefault(lookup_id, {"entity_score": 0.0, "fulltext_score": 0.0})
        merged[lookup_id]["entity_score"] = max(merged[lookup_id]["entity_score"], score)

    for row in fulltext_rows:
        lookup_id = int(row["lookup_id"])
        score = _fts_positive_score(row["bm25_score"])
        merged.setdefault(lookup_id, {"entity_score": 0.0, "fulltext_score": 0.0})
        merged[lookup_id]["fulltext_score"] = max(merged[lookup_id]["fulltext_score"], score)

    ranked = []
    for lookup_id, parts in merged.items():
        # old version was too strict:
        # if entity_terms and parts["entity_score"] == 0:
        #     continue

        hybrid_score = (
            HYBRID_ENTITY_WEIGHT * parts["entity_score"]
            + HYBRID_FULLTEXT_WEIGHT * parts["fulltext_score"]
        )

        ranked.append((lookup_id, hybrid_score, parts["entity_score"], parts["fulltext_score"]))

    ranked.sort(key=lambda x: x[1], reverse=True)
    ranked = ranked[:top_k]

    if not ranked:
        rag_logger.info(
            "Hybrid search empty. entity_terms=%s entity_query=%s fulltext_query_used=%s",
            entity_terms, entity_query, fulltext_query_used
        )
        return []

    ids = [r[0] for r in ranked]
    placeholders = ",".join("?" for _ in ids)
    meta_rows = list(cur.execute(f"""
        SELECT lookup_id, chunk_id, title, subset_name, domain, fulltext_text, source_url
        FROM chunks
        WHERE lookup_id IN ({placeholders})
    """, ids))
    meta_map = {int(r["lookup_id"]): r for r in meta_rows}

    results = []
    for lookup_id, hybrid_score, entity_score, fulltext_score in ranked:
        row = meta_map.get(lookup_id)
        if row is not None:
            results.append(_sqlite_row_to_result(row, hybrid_score, entity_score, fulltext_score))

    rag_logger.info(
        "Hybrid search ok. entity_terms=%s entity_query=%s fulltext_query_used=%s entity_hits=%d fulltext_hits=%d results=%d",
        entity_terms,
        entity_query,
        fulltext_query_used,
        len(entity_rows),
        len(fulltext_rows),
        len(results),
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


def clean_rag_references(docs):
    cleaned = []
    for doc in docs:
        if doc.get("publisher") == TRINEDAY_TOKEN or doc.get("found_on") == TRINEDAY_TOKEN:
            new_doc = dict(doc)
            new_doc["text"] = COPYRIGHT_BLOCK_MSG
            new_doc["snippet"] = COPYRIGHT_BLOCK_MSG
            cleaned.append(new_doc)
        else:
            cleaned.append(doc)
    return cleaned


# ------------------ MAIN ------------------

async def main():
    state = boot()
    refs = await search_references(state, "death squads in Haiti", top_k=10)
    rag_logger.info(json.dumps(refs, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    asyncio.run(main())
