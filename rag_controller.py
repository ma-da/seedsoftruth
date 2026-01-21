"""
Refactored retrieval pipeline
-----------------------------
Key changes vs original:
- No global mutable state
- Explicit RetrievalState object
- Clear stage boundaries (boot, sparse retrieval, routing, shard fetch, rerank, context build)
- Consistent error handling
- Async used only for I/O

"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional, Set
import os
import numpy as np
import requests

import logging_config

# ------------------ CONFIG ------------------
HIVE_RPC = "https://api.hive.blog"
AUTHOR = "wanttoknow"
PERMLINK = "seeds-of-truth-index-registration-0-1"
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
"""
HARDCODED_REGISTRY = {
  "betaslim_manifest.byfile.json": {
    "url_custom": "https://ai.peerservice.org/ipfs/QmVkr8cna15YPiskwuV3emrwAeYC99ALLuRbEjDZK28MnH"
  },
  "groups_urls.json": {
    "url_custom": "https://ai.peerservice.org/ipfs/QmczPZfPVTJbPmrxw8NBwNhdFNy9L7qwfMSaEu1zQvcPsY"
  }
}

USE_HARDCODED_REGISTRY = True

_TOKEN_RE = re.compile(r"[a-z0-9']+")

# retain shards for added speed
_SHARD_CACHE: Dict[str, Any] = {}

# --- Min gating parameters ---

# feature flag for enabling min gating
ENABLE_MIN_GATING = False

# ---------------------------------------------------------------------
# Retrieval distance / sanity gating parameters
#
# These parameters control when a retrieved shard or document is
# considered "relevant enough" to include in RAG context.
#
# They exist to prevent:
#   - accidental matches on rare tokens
#   - long-tail noise from vague queries
#   - flat score distributions where nothing truly stands out
# ---------------------------------------------------------------------

# Minimum cosine similarity between the query vector and a centroid
# (topic cluster) for the shard to be searched at all.
#
# Typical ranges:
#   ~0.02–0.03  : very permissive (high recall, more noise)
#   ~0.04–0.06  : balanced default for mixed corpora
#   ~0.08+      : strict (risk of missing weak but relevant topics)
#
# Below this value, the centroid is effectively unrelated to the query.
_CENTROID_SIM_MIN_GATE = 0.08


# Minimum fraction of query tokens that must also appear in a document
# for it to be considered conceptually relevant.
#
# This guards against documents that match on a single rare word
# but are otherwise about a different topic entirely.
#
# Typical ranges:
#   0.05        : very permissive (allows rare-token matches)
#   0.10–0.15   : balanced (recommended for general RAG)
#   0.20+       : strict (requires strong lexical alignment)
#
# NOTE: This gate penalizes paraphrases (e.g. synonyms) and should
# be looser when embeddings are not used.
_OVERLAP_RATIO_MIN_GATE = 0.20


# Absolute minimum similarity score below which results are always
# rejected, regardless of relative ranking.
#
# This acts as a hard "noise floor" to prevent returning references
# when the entire result set is weak or irrelevant.
#
# Typical ranges:
#   ~0.02–0.03  : permissive
#   ~0.03–0.05  : conservative default
#
# If the best result is below this value, returning nothing is preferred.
_ABS_MIN_MIN_GATE = 0.03


# Relative similarity threshold expressed as a fraction of the best
# scoring result.
#
# This prevents long-tail dilution by keeping only results that are
# "in the same league" as the strongest match for the query.
#
# Typical ranges:
#   0.15        : permissive (keeps broader context)
#   0.25        : balanced default
#   0.40+       : strict (only near-top matches survive)
_REL_FRAC_MIN_GATE = 0.25


# Separation heuristic: how far above the mean (in standard deviations)
# a result must be to be considered meaningfully distinct from background
# noise.
#
# This protects against flat score distributions where all results are
# similarly weak (common with vague or off-corpus queries).
#
# Typical ranges:
#   0.0         : disabled (no separation requirement)
#   0.3–0.5     : balanced default
#   1.0+        : strict outlier-only behavior
#
# Conceptually: require the result to "stand out", not merely rank first.
_Z_CUTOFF_MIN_GATE = 0.5

# --- End min gating parameters ---


rag_logger = logging_config.get_logger("rag")

# ================== STATE ==================
@dataclass
class RetrievalState:
    vocab: List[str]
    token_to_idx: Dict[str, int]
    idf: np.ndarray
    centroids: List[np.ndarray]
    stopwords: Set[str]
    group_list: List[str]
    groups_url_map: Dict[str, str]


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


# ================== BOOT ==================

def hive_get_content(author: str, permlink: str) -> Dict[str, Any]:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "condenser_api.get_content",
        "params": [author, permlink],
    }
    r = requests.post(HIVE_RPC, json=payload)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return data["result"]


def boot() -> RetrievalState:
    """Download registry + artifacts and build immutable retrieval state."""
    rag_logger.info("Booting retrieval system…")

    if USE_HARDCODED_REGISTRY:
        registry = HARDCODED_REGISTRY
    else:
        post = hive_get_content(AUTHOR, PERMLINK)
        # TODO: Fix this. parse_registry_from_post() is not defined in project.
        registry = parse_registry_from_post(post)

    manifest = requests.get(registry["betaslim_manifest.byfile.json"]["url_custom"]).json()
    rag_logger.info("Boot: got manifest")
    
    groups_url_map = requests.get(registry["groups_urls.json"]["url_custom"]).json()
    rag_logger.info("Boot: got groups_url_map")

    files = {f["name"]: f for f in manifest.get("files", [])}

    def file_url(name: str) -> str:
        f = files.get(name)
        if not f:
            raise KeyError(f"Missing file in manifest: {name}")
        return f.get("url_custom") or f.get("url_dedicated")

    vocab = requests.get(file_url("vocabulary.json")).json()
    rag_logger.info("Boot: got vocab")

    idf = np.array(requests.get(file_url("idf.json")).json(), dtype=np.float32)
    centroids = [np.array(row, dtype=np.float32)
                 for row in requests.get(file_url("centroids.json")).json()]
    rag_logger.info("Boot: got centroids")

    stopwords = set(w.lower() for w in requests.get(file_url("stopwords.json")).json())
    rag_logger.info("Boot: got stopwords")

    index_obj = requests.get(
        file_url(manifest.get("roles", {}).get("index", "centroids_index.json"))
    ).json()
    rag_logger.info("Boot: got roles")

    state = RetrievalState(
        vocab=vocab,
        token_to_idx={t: i for i, t in enumerate(vocab)},
        idf=idf,
        centroids=centroids,
        stopwords=stopwords,
        group_list=index_obj["groups"],
        groups_url_map=groups_url_map,
    )

    rag_logger.info(f"✓ Boot complete: {len(vocab):,} vocab | {len(centroids)} centroids")
    return state


# ================== RETRIEVAL ==================

def truncate_question(q: str) -> tuple[str, bool]:
    words = q.split()
    if len(words) <= MAX_QUESTION_WORDS:
        return q.strip(), False
    return " ".join(words[:MAX_QUESTION_WORDS]), True


# --- JS-parity tokenization helpers ---
_STRIP_PHRASES_RE = re.compile(r"(click here|more along these lines|about us)", re.IGNORECASE)
_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

def _strip_phrases(s: str) -> str:
    """JS parity with stripPhrases(): replace known boilerplate phrases with spaces."""
    if not s:
        return ""
    return _STRIP_PHRASES_RE.sub(" ", str(s))

# ----------------------------
# JS-parity query preprocessing
# ----------------------------

# JS uses: t.match(/\b\w{2,}\b/g) || [];
_WORD_RE = re.compile(r"\b\w{2,}\b", re.UNICODE)

# Same stop markers your JS filters for (plus generic </s:...> etc)
_STOP_MARKERS = ["</s>", "<|end|>", "<|eot_id|>"]

def _strip_on_literal_stops(text: str, stops=None) -> str:
    """
    Mirrors JS stripOnLiteralStops(): split once on first stop token,
    allowing optional whitespace/quotes around the token.
    Also handles '</s:1>' style by treating any '</s' as a stop.
    """
    if not text:
        return ""
    s = str(text)

    # Treat any '</s' prefix as a stop (covers </s:1>, </s:2>, etc.)
    # We'll include it as a regex alternative.
    stops = list(stops or _STOP_MARKERS)

    # Escape literal stops for regex
    escaped = [re.escape(x) for x in stops]

    # Add a generic '</s' stopper (covers '</s:1>' etc)
    escaped.append(re.escape("</s"))

    pattern = re.compile(
        r'(?:\s|["\'])*(' + "|".join(escaped) + r')(?:\s|["\'])*',
        flags=re.IGNORECASE
    )

    m = pattern.search(s)
    if not m:
        return s.rstrip()
    return s[:m.start()].rstrip()

def clean_retrieval_text(text: str) -> str:
    """
    Clean *retrieval* text (especially model outputs) so the TF-IDF vector
    isn't dominated by generation scaffolding like '</s:1>Q:' blocks.
    """
    if not text:
        return ""

    s = str(text)

    # 1) Cut off at stop tokens / '</s:1>' etc
    s = _strip_on_literal_stops(s, _STOP_MARKERS)

    # 2) Remove any remaining angle-bracket tags (defensive)
    s = re.sub(r"<[^>]+>", " ", s)

    # 3) Optional: if some models emit repeated "Q:" / "A:" blocks even before </s
    # keep only the first "Note"/summary segment.
    # (This is conservative: only trims if it's clearly a QA template.)
    s = re.sub(r"\b(?:Q:\s*|A:\s*)", " ", s)

    # 4) Your existing phrase stripping (keep your current behavior)
    s = _strip_phrases(s)

    # 5) Normalize whitespace/control chars like the JS pipeline
    s = (
        s.lower()
         .replace("\x00", " ")
    )
    s = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s

def _tokenize_js_parity(text: str) -> List[str]:
    """
    Parity with JS tokenize():
      - strip phrases
      - lowercase
      - collapse whitespace / control chars
      - tokens = \\b\\w{2,}\\b
    """
    t = clean_retrieval_text(text)
    if not t:
        return []
    return _WORD_RE.findall(t)

def build_query_vector(state: RetrievalState, query: str) -> Optional[np.ndarray]:
    query = clean_retrieval_text(query)  # <-- IMPORTANT
    toks = [t for t in _tokenize_js_parity(query) if t not in state.stopwords]
    if not toks:
        return None

    # Build sparse TF map keyed by vocab indices (like JS Map of idx->count)
    tf: Dict[int, int] = {}
    for t in toks:
        idx = state.token_to_idx.get(t)
        if idx is None:
            continue
        tf[idx] = tf.get(idx, 0) + 1

    if not tf:
        return None

    qv = np.zeros(len(state.vocab), dtype=np.float32)
    # qv[i] = tf * idf
    for idx, f in tf.items():
        qv[idx] = float(f) * float(state.idf[idx])

    # normalize
    norm = float(np.linalg.norm(qv)) + 1e-12
    qv /= norm
    return qv


def top_k_centroids(
    state: RetrievalState,
    qv: np.ndarray,
    k: int = 9,
    min_sim: float = 0.04,  # NEW: centroid similarity gate
) -> List[int]:
    """
    Returns top-k centroid indices whose cosine similarity to qv
    exceeds a minimum threshold.
    """

    # cosine similarity because qv and centroids are normalized
    sims = [float(np.dot(qv, c)) for c in state.centroids]

    # rank by similarity
    ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)

    # NEW: gate weak centroids
    gated = [i for i in ranked if sims[i] >= min_sim]

    rag_logger.info(f"top_k_centroids distance gating excluded {len(ranked)-len(gated)} number of references, min_sim={min_sim}")
    if gated:
        best = sims[gated[0]]
        worst = sims[gated[-1]]

        rag_logger.info(
            f"[centroids] accepted={len(gated)} | "
            f"best={best:.4f} | worst={worst:.4f} | "
            f"threshold={min_sim:.4f}"
        )

    if len(gated):
        rag_logger.info(f"top_k_centroids best distance {len(gated)} number of references")

    return gated[:k]


async def fetch_json(url: str) -> Any:
    if url in _SHARD_CACHE:
        return _SHARD_CACHE[url]
    loop = asyncio.get_running_loop()
    r = await loop.run_in_executor(None, requests.get, url)
    r.raise_for_status()
    data = r.json()
    _SHARD_CACHE[url] = data
    return data


async def sparse_retrieve(
    state: RetrievalState,
    query: str,
    *,
    centroid_k: int = 20,
    max_per_shard: int | None = None,
) -> List[Dict[str, Any]]:
    qv = build_query_vector(state, query)
    if qv is None:
        return []

    cent_ids = top_k_centroids(state, qv, k=centroid_k)
    shard_names = [state.group_list[i] for i in cent_ids]

    shards = await asyncio.gather(*[fetch_json(state.groups_url_map[name]) for name in shard_names])

    scored: List[Dict[str, Any]] = []
    for shard in shards:
        rows = shard if (max_per_shard is None) else shard[:max_per_shard]
        for row in rows:
            sim = sum(weight * qv[idx] for idx, weight in row.get("tfidf", []))
            norm = row.get("norm", 1.0)
            score = sim / norm if norm > 0 else 0.0

            scored.append({
                "row_id": row.get("row_id"),
                "title": row.get("title", ""),
                "text": row.get("text", ""),
                "source": row.get("source", ""),
                "score": float(score),   # <- use consistent key
            })

    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored



# ================== CONTEXT ==================

_SENT_RE = re.compile(r"[^.!?]+[.!?]*")

def _sentence_split(text: str) -> List[str]:
    s = re.sub(r"\s+", " ", text or "").strip()
    if not s:
        return []
    return [m.group(0).strip() for m in _SENT_RE.finditer(s) if m.group(0).strip()]

def _to_term_set(query: str) -> Set[str]:
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
    """
    JS-parity compression with trigram de-dup.
    Outputs <doc> blocks similar to your JS buildCompressedContext.
    """
    terms = _to_term_set(query)
    global_tris: Set[str] = set()
    blocks: List[str] = []

    picked = (docs or [])[: int(context_k)]
    for d in picked:
        text = (d.get("text") or d.get("snippet") or "").strip()
        if not text:
            continue

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
            # keep if introduces any new trigram
            introduce = any(tri not in global_tris for tri in tris)
            if introduce:
                kept.append(s.strip())
                for tri in tris:
                    global_tris.add(tri)

        if not kept:
            continue

        doc_id = d.get("row_id") or d.get("title") or ""
        url = d.get("source") or ""
        score = d.get("score_rerank")
        if score is None:
            score = d.get("score_tfidf") or 0.0

        # NOTE: keep it compact (your SYSTEM_PROMPT forbids lists, but internal docs can contain bullets)
        block = (
            f'<doc id="{doc_id}" url="{url}" score="{float(score):.2f}">\n'
            "<snippets>\n- " + "\n- ".join(kept) + "\n</snippets>\n"
            "</doc>"
        )
        blocks.append(block)

    return "\n".join(blocks)


# ================== ORCHESTRATION ==================

HF_ENDPOINT_URL = "https://cr41uamktrsdyg3d.us-east-1.aws.endpoints.huggingface.cloud"

# The Huggingface token
# NOTE DO NOT CHECK IN THE ACTUAL HF TOKEN
HF_API_KEY = os.getenv("HF_API_KEY", "").strip()

HF_TIMEOUT = int(os.getenv("HF_TIMEOUT_SECS", "900"))
HF_MAX_ATTEMPTS = int(os.getenv("HF_MAX_ATTEMPTS", "10"))
HF_MAX_WAIT_SECS = int(os.getenv("HF_MAX_WAIT_SECS", "6"))

# Optional: custom warmup prompt (keeps your esoteric style)
HF_WARMUP_PROMPT = "Q: [warmup] A:"
HF_WARMUP_MAX_NEW_TOKENS = 16

# headers sent in endpoint model requests
HF_REQ_HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json",
}

HF_MAX_ALLOWED_NEW_TOKENS = 1200

# ask message payload
HF_PAYLOAD = {"inputs": "payload"}

# health check payload
HF_HEALTH_PAYLOAD = {"inputs": "health_check"}

# poll interval used by background workers
WORKER_POLL_INTERVAL_SECS = 2

# the number of workers to use in the thread pool.
# usually this should be set to max of the number of model instance replicas
MAX_WORKERS = 1

# this queue holds the jobs awaiting to be processed
job_queue = []
job_lock = threading.Lock()

# this queue holds outgoing responses that need to be sent to client
outgoing_queue = []
outgoing_lock = threading.Lock()

# this track the users in-flights so we can quickly check if there is an inflight request
# currently we will only allow one inflight request at a time
inflight_users = {}
inflight_lock = threading.Lock()

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
    """
    Parse common HF Inference Endpoint response formats.
    Matches your JS logic:
      - [{generated_text: "..."}]
      - {generated_text: "..."}
      - {choices: [{text: "..."}]} or {choices: [{message:{content:"..."}}]}
      - fallback to string/json
    """
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
    """
    Sync HF call with 503 model-loading retry. Raises RuntimeError with details on failure.
    """
    if not HF_ENDPOINT_URL:
        rag_logger.error("Missing HF_ENDPOINT_URL")
        raise RuntimeError("Missing HF_ENDPOINT_URL")

    if not HF_API_KEY:
        rag_logger.error("Missing HF_API_KEY")
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

        # HF cold-start/model loading
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

        # Non-OK: raise with body to debug schema/auth/etc
        if not r.ok:
            body = (r.text or "")[:2000]
            raise RuntimeError(f"HF error {r.status_code}: {body}")

        # OK response
        try:
            data = r.json()
        except Exception:
            raise RuntimeError(f"HF returned non-JSON: {(r.text or '')[:2000]}")

        return _parse_hf_text(data).strip()

    raise RuntimeError(f"HF model still loading (503). Last: {last_detail or 'n/a'}")


async def ask(state: "RetrievalState", question: str, *, context_k: int = 5, centroid_k: int = 20, verbose: bool = True) -> str:
    rag_logger.info(f"rag_controller begin question ask, prompt: {question[:40]}...")
    q, truncated = truncate_question(question)

    if verbose:
        rag_logger.info("Searching corpus…")
    t0 = time.time()

    docs = await sparse_retrieve(state, q, centroid_k=centroid_k, max_per_shard=None)
    context = build_context(docs, q, context_k=context_k)

    if verbose:
        rag_logger.info(f"Retrieved context in {time.time() - t0:.2f}s")

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Agent Question:\n"
        f"{q}\n\n"
        "DOCUMENTS (internal — do not mention they exist):\n"
        f"{context}"
    )

    answer = _hf_generate(prompt, temperature=0.3, max_new_tokens=768)

    if truncated:
        answer = "(Question truncated)\n\n" + answer

    rag_logger.info(f"rag_controller finished question ask, answer: {answer[:40]}...")
    return answer.strip()


# ---- shard cache ----

def _sparse_dot(tfidf_pairs: List[List[Any]], qv: np.ndarray) -> float:
    """
    Matches JS sparseDot(r.tfidf, qv): sum(weight * qv[idx])
    tfidf_pairs is list of [idx, weight]
    """
    s = 0.0
    for p in tfidf_pairs or []:
        try:
            idx = int(p[0])
            w = float(p[1])
        except Exception:
            continue
        s += w * float(qv[idx])
    return float(s)

_URL_RE = re.compile(r"(https?://[^\s<]+)")

def _linkify_plain_urls(html_text: str) -> str:
    return _URL_RE.sub(r'<a href="\1" target="_blank" rel="noopener noreferrer">\1</a>', html_text)

def _truncate_words(text: str, max_words: int) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return text or ""
    return " ".join(words[:max_words]) + "…"

def _snippet_html_from_text(text: str, max_words: int = 500) -> str:
    # treat dataset text as plain text (safe). If your corpus contains safe HTML, adjust.
    import html as _html
    t = _truncate_words(text or "", max_words)
    t = _html.escape(t).replace("\n", "<br>")
    return _linkify_plain_urls(t)

async def search_references(
    state: RetrievalState,
    query: str,
    *,
    top_k: int = 10,
    centroid_k: int = 20,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    JS-parity datasetSearch():
      - qv from tokenize() / stopwords / tf*idf / normalize
      - topKCentroids(qv, centroid_k)
      - fetch those shards
      - score every row: sparseDot(row.tfidf,qv)/row.norm
      - sort desc, return top_k
    """
    if not state.group_list:
        raise RuntimeError("Index not loaded (group_list empty).")

    qv = build_query_vector(state, query)
    if qv is None:
        return {"query": query, "num_results": 0, "results": [], "message": "No indexable terms."}

    if ENABLE_MIN_GATING:
        min_sim = _CENTROID_SIM_MIN_GATE
    else:
        min_sim = 0.0

    cent_ids = top_k_centroids(state, qv, k=int(centroid_k), min_sim=min_sim)
    shard_names = [state.group_list[i] for i in cent_ids]

    # Validate shard URLs to avoid KeyError surprises
    urls: List[str] = []
    for name in shard_names:
        url = state.groups_url_map.get(name)
        if not url:
            raise RuntimeError(f"No URL for shard '{name}' in groups_url_map.")
        urls.append(url)

    shards = await asyncio.gather(*[fetch_json(u) for u in urls])

    # Precompute query sparsity for overlap gating
    qv_nonzero_idxs = set(np.nonzero(qv)[0])

    num_shards_skipped = 0
    scored: List[Dict[str, Any]] = []
    for shard in shards:
        # shard is a list of rows
        for row in shard or []:
            tfidf_pairs = row.get("tfidf") or []
            norm = float(row.get("norm") or 1.0)
            sim = _sparse_dot(tfidf_pairs, qv) / norm if norm > 0 else 0.0

            # token overlap sanity check
            # if shared contains too few of the query's token, discard it,
            # no matter what the similarity scores says.
            row_idxs = {int(p[0]) for p in tfidf_pairs if len(p) >= 2}
            overlap_ratio = (
                    len(row_idxs & qv_nonzero_idxs) /
                    max(1, len(qv_nonzero_idxs))
            )

            if ENABLE_MIN_GATING and overlap_ratio < _OVERLAP_RATIO_MIN_GATE:
                num_shards_skipped += 1
                continue

            scored.append({
                "row_id": row.get("row_id"),
                "source": row.get("source") or "",
                "title": row.get("title") or "",
                "text": row.get("text") or "",
                "snippet": (row.get("text") or "")[:1200],   # JS parity
                "snippet_html": _snippet_html_from_text(row.get("text") or "", max_words=500),
                "score_tfidf": float(sim),                  # JSON-safe
            })
    rag_logger.info(f"Skipped {num_shards_skipped} rows in shards due to overlap ratios")

    if not scored:
        return {
            "query": query,
            "num_results": 0,
            "results": [],
            "message": "No relevant rows after gating."
        }

    # adaptive distance gating (absolute + relative)
    scores = np.array(
        [r["score_tfidf"] for r in scored],
        dtype=np.float32,
    )

    if ENABLE_MIN_GATING:
        max_score = float(scores.max())
        mean_score = float(scores.mean())
        std_score = float(scores.std())

        threshold1 = max_score * _REL_FRAC_MIN_GATE
        threshold2 = mean_score + _Z_CUTOFF_MIN_GATE * std_score

        threshold = max(
            _ABS_MIN_MIN_GATE,
            threshold1,
            threshold2
        )

        rag_logger.info(
            f"Threshold calculation: value {threshold}, abs_min_gate {_ABS_MIN_MIN_GATE}, rel_frac {threshold1}, z_cutoff {threshold2}")
    else:
        threshold = 0

    # Apply gating
    scored = [
        r for r in scored
        if r["score_tfidf"] >= threshold
    ]

    if not scored:
        return {
            "query": query,
            "num_results": 0,
            "results": [],
            "message": "No sufficiently relevant references found."
        }

    scored.sort(key=lambda d: d["score_tfidf"], reverse=True)
    results = scored[:int(top_k)]

    return {
        "query": query,
        "num_results": len(results),
        "results": results,
        "message": f"Found {len(results)} result(s).",
    }

# ================== Queueing ==================

def queue_job(user_id, job_id, msg) -> bool:
    global job_queue, job_lock

    queued_job = QueuedJob(user_id, job_id, msg)
    with job_lock:
        job_queue.append(queued_job)
        qsize = len(job_queue)

    rag_logger.info(f"Job with id {job_id} was queued successfully. Curr depth: {qsize}")
    return qsize


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


# ================== MAIN ==================

async def main():
    state = boot()

    # Test ask method
    #answer = await ask(
    #    state,
    #    "What really happened on 9/11 according to declassified documents and whistleblowers?",
    #)
    #rag_logger.info("\nANSWER:\n" + "=" * 40)
    #rag_logger.info(answer)

    # Test search_references method
    refs = await search_references(
        state,
        "What do declassified documents say about JFK assassination planning?",
        top_k=10,
        verbose=True,
    )
    rag_logger.info(json.dumps(refs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())