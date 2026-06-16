"""rag_context.py — prompt-context assembly from retrieved chunks.

Extracted from rag_controller.py (2026-05-20 refactor). ``build_context_v2``
takes the chunks returned by ``rag_retrieval.search_references`` and
assembles the context block that is handed to the LLM.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import logging_config
import rag_cleaner
from rag_retrieval import (
    _QUERY_STOPWORDS,
    RetrievalState,
    _sentence_split,
    _to_term_set,
    _tokenize_simple,
    _trigrams,
    extract_canonical_entity_terms,
)

rag_logger = logging_config.get_logger("rag")


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
            rag_logger.warning(
                f"build_context_v2: entity extraction failed: {e}"
            )

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
        rag_logger.info(
            f"build_context_v2: no docs survived filter (of {len(picked)} candidates)"
        )
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
