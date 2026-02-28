"""
rag_controller.py (BM25S edition)
- Loads Trine Day Mini chunks from local disk
- Retrieves with bm25s (single index or shard folder)
- Keeps HF generation + queueing logic (minimal changes)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import bm25s

import logging_config
import rag_cleaner

rag_logger = logging_config.get_logger("rag")

# ------------------ CONFIG ------------------

MAX_QUESTION_WORDS = 400
TOP_DOCS = 5

SYSTEM_PROMPT = """
Answer the question in 1–3 concise paragraphs (total <300 words).
Use proper spelling, punctuation, and spacing.
Do not run words together.
Avoid long strings of numbers.
NO LISTS. If unavoidable, then limit lists to 3 items maximum.
Focus only on the question asked, avoiding unrelated topics or meta-text (e.g., "Note:", "click here").
Do not suggest other references, "further reading", or "Note:" for the reader to explore, view, or to learn more.
Stop after the answer.
/no_think
""".strip()

# --- Data paths (override via env) ---
TRINEDAY_DATA_DIR = Path(os.getenv("TRINEDAY_DATA_DIR", "./data/trineday_mini"))
CHUNKS_JSONL = Path(os.getenv("TRINEDAY_CHUNKS_JSONL", str(TRINEDAY_DATA_DIR / "trineday_mini_chunks.jsonl")))

# Either single-index mode :
BM25_INDEX_DIR = Path(os.getenv("TRINEDAY_BM25_INDEX_DIR", str(TRINEDAY_DATA_DIR / "bm25_index")))

# Or shard mode:
BM25_SHARDS_DIR = Path(os.getenv("TRINEDAY_BM25_SHARDS_DIR", str(TRINEDAY_DATA_DIR / "shards")))

# Retrieval tuning
DEFAULT_TOP_K = int(os.getenv("TRINEDAY_TOP_K", "10"))
# in shard mode, pull this many from each shard then merge
PER_SHARD_K = int(os.getenv("TRINEDAY_PER_SHARD_K", "30"))

# Tokenization options
BM25_STOPWORDS = os.getenv("TRINEDAY_BM25_STOPWORDS", "en")  # "en" or None
BM25_USE_STEMMER = os.getenv("TRINEDAY_BM25_STEMMER", "0") == "1"  # keep off by default

# --- Min gating parameters ---

# feature flag for enabling min gating
ENABLE_MIN_GATING = False

# ------------------ STATE ------------------

@dataclass
class BM25Shard:
    name: str
    index: bm25s.BM25
    # doc_idx in the global dataframe that this shard contains, in same order as indexed corpus
    doc_idx: np.ndarray

@dataclass
class RetrievalState:
    df: pd.DataFrame
    # exactly one of these is used:
    bm25: Optional[bm25s.BM25]
    shards: Optional[List[BM25Shard]]

@dataclass
class QueuedJob:
    user_id: str
    job_id: str
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


# ------------------ BOOT ------------------

def _load_df(jsonl_path: Path) -> pd.DataFrame:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing dataset JSONL: {jsonl_path}")

    df = pd.read_json(jsonl_path, lines=True)

    # Ensure required columns exist
    for col in ["title", "text", "source_url", "publisher", "found_on"]:
        if col not in df.columns:
            df[col] = ""

    # Ensure doc_idx is contiguous
    if "doc_idx" not in df.columns:
        df.insert(0, "doc_idx", range(len(df)))

    df = df.sort_values("doc_idx").reset_index(drop=True)
    df["doc_idx"] = range(len(df))
    return df

def _load_single_bm25(index_dir: Path) -> bm25s.BM25:
    if not index_dir.exists() or not any(index_dir.iterdir()):
        raise FileNotFoundError(
            f"BM25 index dir not found or empty: {index_dir}\n"
            f"Set TRINEDAY_BM25_INDEX_DIR or upload your bm25_index folder."
        )
    # load_corpus=False keeps memory down
    return bm25s.BM25.load(str(index_dir), load_corpus=False)

def _load_shards(shards_dir: Path) -> List[BM25Shard]:
    if not shards_dir.exists():
        raise FileNotFoundError(
            f"BM25 shards dir not found: {shards_dir}\n"
            f"Set TRINEDAY_BM25_SHARDS_DIR or upload your shards folder."
        )

    shard_dirs = sorted([p for p in shards_dir.iterdir() if p.is_dir()])
    if not shard_dirs:
        raise FileNotFoundError(f"No shard directories found under: {shards_dir}")

    shards: List[BM25Shard] = []
    for sd in shard_dirs:
        # Convention:
        #  shard_xxx/bm25/   (bm25s saved index)
        #  shard_xxx/doc_idx.npy  (global doc_idx list)
        bm25_dir = sd / "bm25"
        doc_idx_path = sd / "doc_idx.npy"

        if not bm25_dir.exists():
            raise FileNotFoundError(f"Missing shard bm25 directory: {bm25_dir}")
        if not doc_idx_path.exists():
            raise FileNotFoundError(f"Missing shard doc_idx.npy: {doc_idx_path}")

        idx = bm25s.BM25.load(str(bm25_dir), load_corpus=False)
        doc_idx = np.load(str(doc_idx_path))
        shards.append(BM25Shard(name=sd.name, index=idx, doc_idx=doc_idx))

    return shards

def boot() -> RetrievalState:
    rag_logger.info("Booting BM25 retrieval…")

    df = _load_df(CHUNKS_JSONL)
    rag_logger.info(f"Boot: loaded chunks df rows={len(df):,} from {CHUNKS_JSONL}")

    # Prefer shard mode if shards exist and are non-empty
    shards = None
    bm25 = None

    try:
        if BM25_SHARDS_DIR.exists() and any(BM25_SHARDS_DIR.iterdir()):
            shards = _load_shards(BM25_SHARDS_DIR)
            rag_logger.info(f"Boot: loaded {len(shards)} BM25 shards from {BM25_SHARDS_DIR}")
        else:
            bm25 = _load_single_bm25(BM25_INDEX_DIR)
            rag_logger.info(f"Boot: loaded single BM25 index from {BM25_INDEX_DIR}")
    except Exception as e:
        raise RuntimeError(f"BM25 boot failed: {e}") from e

    return RetrievalState(df=df, bm25=bm25, shards=shards)


# ------------------ TEXT CLEANING / TOKENIZE ------------------

_STRIP_PHRASES_RE = re.compile(r"(click here|more along these lines|about us)", re.IGNORECASE)
_STOP_MARKERS = ["</s>", "<|end|>", "<|eot_id|>"]
_WORD_RE = re.compile(r"\b\w{2,}\b", re.UNICODE)

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
    pattern = re.compile(r'(?:\s|["\'])*(' + "|".join(escaped) + r')(?:\s|["\'])*', flags=re.IGNORECASE)
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

def _bm25_tokenize_query(query: str):
    # bm25s.tokenize accepts either a string or list[str] depending on version;
    # safest is passing a list and taking first.
    q = clean_retrieval_text(query)
    toks = bm25s.tokenize([q], stopwords=(BM25_STOPWORDS if BM25_STOPWORDS else None))
    # toks is list-of-token-lists
    return toks


# ------------------ RETRIEVAL ------------------

_URL_RE = re.compile(r"(https?://[^\s<]+)")

def _linkify_plain_urls(html_text: str) -> str:
    return _URL_RE.sub(r'<a href="\1" target="_blank" rel="noopener noreferrer">\1</a>', html_text)

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

def _row_to_result(df: pd.DataFrame, row: pd.Series, score: float) -> Dict[str, Any]:
    txt = str(row.get("text") or "")
    src = str(row.get("source_url") or "").strip()

    return {
        "row_id": row.get("chunk_id") or row.get("doc_idx"),

        "source_url": src,
        "source": src,

        "title": row.get("title") or "",
        "publisher": row.get("publisher") or "",
        "found_on": row.get("found_on") or "",
        "text": txt,
        "snippet": txt[:1200],
        "snippet_html": _snippet_html_from_text(txt, max_words=600),
        "score_bm25": float(score),
    }

def _search_single_index(state: RetrievalState, query: str, top_k: int) -> List[Dict[str, Any]]:
    assert state.bm25 is not None
    q_tokens = _bm25_tokenize_query(query)

    t0 = time.perf_counter()
    doc_ids, scores = state.bm25.retrieve(q_tokens, k=top_k)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    doc_ids = doc_ids[0]
    scores = scores[0]
    results: List[Dict[str, Any]] = []

    for did, sc in zip(doc_ids, scores):
        if hasattr(did, "item"):
            did = did.item()
        if hasattr(sc, "item"):
            sc = sc.item()
        # did is doc_idx row index (0..N-1) if you indexed in df order
        row = state.df.iloc[int(did)]
        results.append(_row_to_result(state.df, row, float(sc)))

    rag_logger.info(f"BM25 single-index search: top_k={top_k} time_ms={dt_ms:.1f}")
    return results

def _search_shards(state: RetrievalState, query: str, top_k: int) -> List[Dict[str, Any]]:
    assert state.shards is not None
    q_tokens = _bm25_tokenize_query(query)

    t0 = time.perf_counter()

    candidates: List[Tuple[float, int]] = []  # (score, global_doc_idx)

    for sh in state.shards:
        doc_ids, scores = sh.index.retrieve(q_tokens, k=min(PER_SHARD_K, top_k))
        doc_ids = doc_ids[0]
        scores = scores[0]
        for did, sc in zip(doc_ids, scores):
            if hasattr(did, "item"):
                did = did.item()
            if hasattr(sc, "item"):
                sc = sc.item()
            # did is local index inside shard => map to global doc_idx
            global_doc_idx = int(sh.doc_idx[int(did)])
            candidates.append((float(sc), global_doc_idx))

    # merge: take best score per global doc (dedupe)
    best: Dict[int, float] = {}
    for sc, gidx in candidates:
        if (gidx not in best) or (sc > best[gidx]):
            best[gidx] = sc

    merged = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    dt_ms = (time.perf_counter() - t0) * 1000.0

    results: List[Dict[str, Any]] = []
    for gidx, sc in merged:
        row = state.df.iloc[int(gidx)]
        results.append(_row_to_result(state.df, row, float(sc)))

    rag_logger.info(
        f"BM25 shard search: shards={len(state.shards)} per_shard_k={PER_SHARD_K} "
        f"candidates={len(candidates)} merged={len(results)} time_ms={dt_ms:.1f}"
    )
    return results

async def search_references(
    state: RetrievalState,
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    verbose: bool = False,
    **_unused,
) -> Dict[str, Any]:
    """
    Replacement for TF-IDF shard-based search_references().
    Returns the same shape your frontend expects: {query,num_results,results,message}
    """
    q = (query or "").strip()
    if not q:
        return {"query": query, "num_results": 0, "results": [], "message": "Empty query."}

    if state.shards is not None:
        results = _search_shards(state, q, top_k=int(top_k))
    else:
        results = _search_single_index(state, q, top_k=int(top_k))

    return {
        "query": query,
        "num_results": len(results),
        "results": results,
        "message": f"Found {len(results)} result(s).",
    }

# ------------------ CONTEXT BUILD (kept) ------------------

_SENT_RE = re.compile(r"[^.!?]+[.!?]*")

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
    i = 0
    for d in picked:
        text = (d.get("text") or d.get("snippet") or "").strip()
        if not text:
            continue
        i = i + 1
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
    window_radius: int = 1,  # include ±1 neighboring sentence
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

        # ----------------------------
        # 1️⃣ Score all sentences
        # ----------------------------

        scored_indices = []

        for idx, s in enumerate(sents):
            low = s.lower()
            sent_tokens = set(_tokenize_simple(low))

            overlap = len(terms & sent_tokens)
            if overlap == 0:
                continue

            # simple score: term overlap count
            score = overlap

            scored_indices.append((idx, score))

        if not scored_indices:
            continue

        # Sort by descending score
        scored_indices.sort(key=lambda x: x[1], reverse=True)

        # Keep top sentence indices
        top_indices = [idx for idx, _ in scored_indices[:max_snips_per_doc]]

        # ----------------------------
        # 2️⃣ Expand windows
        # ----------------------------

        windows = []

        for idx in top_indices:
            start = max(0, idx - window_radius)
            end = min(len(sents), idx + window_radius + 1)
            windows.append((start, end))

        # ----------------------------
        # 3️⃣ Merge overlapping windows
        # ----------------------------

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

        # ----------------------------
        # 4️⃣ Build kept text with trigram diversity
        # ----------------------------

        kept_blocks: List[str] = []

        for start, end in merged:
            window_text = " ".join(sents[start:end]).strip()

            toks = _tokenize_simple(window_text)
            tris = _trigrams(toks)

            # Check if window introduces novelty
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


# ------------------ HF ORCHESTRATION (unchanged from your version) ------------------

# endpoint wtk-trineday-mini-llama3-70b-muu
HF_ENDPOINT_URL = "https://d6pfgv6yisy4pld2.us-east-1.aws.endpoints.huggingface.cloud" #os.getenv("HF_ENDPOINT_URL", "").strip()

HF_API_KEY = os.getenv("HF_API_KEY", "").strip()
HF_TIMEOUT = int(os.getenv("HF_TIMEOUT_SECS", "900"))
HF_MAX_ATTEMPTS = int(os.getenv("HF_MAX_ATTEMPTS", "10"))
HF_MAX_WAIT_SECS = int(os.getenv("HF_MAX_WAIT_SECS", "6"))
HF_WARMUP_PROMPT = "Q: [warmup] A:"
HF_WARMUP_MAX_NEW_TOKENS = 16
HF_MAX_ALLOWED_NEW_TOKENS = 1200

HF_REQ_HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json",
}

HF_HEALTH_PAYLOAD = {"inputs": "health_check"}
HF_REQ_TIMEOUT_SECS = 5


async def is_model_ready(timeout=HF_REQ_TIMEOUT_SECS) -> bool:
    rag_logger.info("checking health...")
    try:
        r = requests.post(f"{HF_ENDPOINT_URL}",
                          headers=HF_REQ_HEADERS,
                          json=HF_HEALTH_PAYLOAD,
                          timeout=timeout)
        if r.status_code == 200:
            if r.json().get("health") == "ok":
                rag_logger.info("Model ready: Custom health response received")
                return True
            else:
                rag_logger.info("Processed response but not explicit health OK")
                return False  # Or False if strict
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


async def send_warmup() -> bool:
    payload = {
        "inputs": HF_WARMUP_PROMPT,
        "parameters": {
            "max_new_tokens": HF_WARMUP_MAX_NEW_TOKENS,
            "temperature": 0.1,
            "stop_sequences": ["\n", "Q:"]
        }
    }
    try:
        r = requests.post(
            f"{HF_ENDPOINT_URL}/generate",
            json=payload,
            headers=HF_REQ_HEADERS,
            timeout=60
        )
        if r.status_code == 200:
            rag_logger.info(f"Warm-up successful! Response: {r.json().get('generated_text', '')[:100]}")
            return True
    except Exception as e:
        rag_logger.warning(f"Warm-up request failed: {e}")

    return False


def _parse_hf_text(data) -> str:
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

def _hf_generate(prompt: str, *, temperature: float, max_new_tokens: int) -> str:
    if not HF_ENDPOINT_URL:
        raise RuntimeError("Missing HF_ENDPOINT_URL")
    if not HF_API_KEY:
        raise RuntimeError("Missing HF_API_KEY")

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
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

    for attempt in range(1, HF_MAX_ATTEMPTS + 1):
        r = requests.post(HF_ENDPOINT_URL, headers=headers, json=payload, timeout=HF_TIMEOUT)

        if r.status_code == 503:
            try:
                j = r.json()
            except Exception:
                j = {}
            wait = int(j.get("estimated_time") or 3)
            wait = max(1, min(HF_MAX_WAIT_SECS, wait))
            time.sleep(wait)
            last_detail = f"503 loading; wait={wait}s; attempt={attempt}/{HF_MAX_ATTEMPTS}"
            continue

        if not r.ok:
            body = (r.text or "")[:2000]
            raise RuntimeError(f"HF error {r.status_code}: {body}")

        try:
            data = r.json()
        except Exception:
            raise RuntimeError(f"HF returned non-JSON: {(r.text or '')[:2000]}")

        return _parse_hf_text(data).strip()

    raise RuntimeError(f"HF model still loading (503). Last: {last_detail or 'n/a'}")

async def ask(state: RetrievalState,
              question: str,
              *,
              context_k: int = 5,
              top_k: int = 10,
              verbose: bool = True,
              use_double_prompt = False) -> str:
    rag_logger.info(f"ask(): {question[:80]}...")
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
            f"{SYSTEM_PROMPT}\n\n"
            "Agent Question:\n"
            f"{q}\n"
            f"{q}\n\n"
            "DOCUMENTS (internal — do not mention they exist):\n"
            f"{context}"
        )
    else:
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "Agent Question:\n"
            f"{q}\n\n"
            "DOCUMENTS (internal — do not mention they exist):\n"
            f"{context}"
        )

    rag_logger.info(f"-- PROMPT --\n{prompt} \n-- END PROMPT --")

    answer = _hf_generate(prompt, temperature=0.3, max_new_tokens=768)

    if truncated:
        answer = "(Question truncated)\n\n" + answer

    return answer.strip()

async def ask_model_only(
    state: RetrievalState,
    question: str,
    *,
    verbose: bool = True,
    use_double_prompt = False,
) -> str:
    """
    Model-only answer: NO retrieval, NO documents injected.
    Still applies truncate_question and uses the same SYSTEM_PROMPT.
    """
    rag_logger.info(f"ask_model_only(): {question[:80]}...")
    q, truncated = truncate_question(question)

    if use_double_prompt:
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "Agent Question:\n"
            f"{q}\n"
            f"{q}\n"
        )
    else:
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "Agent Question:\n"
            f"{q}\n"
        )

    rag_logger.info(f"-- PROMPT --\n{prompt} \n-- END PROMPT --")

    if verbose:
        rag_logger.info(f"-- PROMPT (model-only) --\n{prompt}\n-- END PROMPT --")

    answer = _hf_generate(prompt, temperature=0.3, max_new_tokens=768)

    if truncated:
        answer = "(Question truncated)\n\n" + answer

    return str(answer or "").strip()

# ------------------ Queueing (kept) ------------------

job_queue = []
job_lock = threading.Lock()

outgoing_queue = []
outgoing_lock = threading.Lock()

inflight_users = {}
inflight_lock = threading.Lock()

def queue_job(user_id, job_id, msg) -> bool:
    queued_job = QueuedJob(user_id, job_id, msg)
    with job_lock:
        job_queue.append(queued_job)
        return len(job_queue)


def fetch_queued_job_info(user_id) -> (int, Optional[QueuedJob]):
    global job_queue, job_lock

    with job_lock:
        for i, item in enumerate(job_queue):
           if user_id == item.user_id:
               return i, item

    return 0, None


def has_queued_job() -> bool:
    global job_queue, job_lock

    with job_lock:
        return len(job_queue) > 0


def get_next_queued_job() -> Optional[QueuedJob]:
    global job_queue, job_lock

    with job_lock:
        if len(job_queue) == 0:
            return None
        return job_queue[0]


def pop_next_queued_job():
    global job_queue, job_lock

    with job_lock:
        if len(job_queue) == 0:
            return
        job_queue.pop(0)


def job_queue_len():
    global job_queue, job_lock

    with job_lock:
        return len(job_queue)


def queue_outgoing(response) -> bool:
    global outgoing_queue, outgoing_lock

    if not isinstance(response, QueuedResponse):
        raise RuntimeError("queue_outgoing requires QueuedResponse param")

    with outgoing_lock:
        outgoing_queue.append(response)
        qsize = len(outgoing_queue)

    rag_logger.info(f"Outgoing resp with job_id {response.job_id} was queued successfully for sending. Curr depth: {qsize}")
    return qsize


# Currently, we only support one inflight request. This might have to change in the future if we do more.
# Fetching removes the item from queue
def fetch_queued_outgoing_info(user_id) -> Optional[QueuedResponse]:
    global outgoing_queue, outgoing_lock

    with outgoing_lock:
        for i, item in enumerate(outgoing_queue):
            #rag_logger.info(f"Outgoing fetch: found {item.user_id} vs. target {user_id}")
            if user_id == item.user_id:
                del outgoing_queue[i]
                rag_logger.info(f"Fetched queue outgoing info for job_id {item.job_id}, user_id {item.user_id}")
                return item
            else:
                rag_logger.info("Queued msg did not match")

    return None


def outgoing_queue_len():
    global outgoing_queue, outgoing_lock

    with outgoing_lock:
        return len(outgoing_queue)


# ------------------ MAIN (for local CLI test) ------------------

async def main():
    state = boot()
    refs = await search_references(state, "death squads in Haiti", top_k=10)
    rag_logger.info(json.dumps(refs, indent=2, ensure_ascii=False)[:4000])

if __name__ == "__main__":
    asyncio.run(main())
