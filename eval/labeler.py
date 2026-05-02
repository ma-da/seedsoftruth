#!/usr/bin/env python3
"""
LLM-judge labeler for the A/B retrieval eval set.

Reads queries_seed.jsonl, runs each (non-probe) query through /api/search
once per variant at large top-k, pools the unique candidate docs, and asks
an LLM to grade each (query, doc) pair on a 0–3 scale. Caches judgments so
re-runs are nearly free. Writes labels.jsonl in the format ab_eval.py expects.

USAGE
  # Anthropic (default)
  export ANTHROPIC_API_KEY=sk-ant-...
  python scripts/eval/labeler.py --judge anthropic \
      --queries scripts/eval/queries_seed.jsonl \
      --base-url http://localhost:5000 \
      --output scripts/eval/labels.jsonl

  # DeepInfra (Llama-3.1-70B-Instruct by default)
  export DEEPINFRA_API_KEY=...
  python scripts/eval/labeler.py --judge deepinfra \
      --queries scripts/eval/queries_seed.jsonl \
      --base-url http://localhost:5000 \
      --output scripts/eval/labels.jsonl

  # Stub (no key needed, fake grades — pipeline test only)
  python scripts/eval/labeler.py --judge stub --queries ... --output ...

OUTPUT FILES
  labels.jsonl       — one record per query, ab_eval-compatible:
                       in-domain: {"query": q, "labels": {row_id: grade}, "id_field": "row_id"}
                       probe:     {"query": q, "probe": true}
  judge_cache.jsonl  — append-only cache of (query, doc_id, grade, rationale).
                       Re-running with the same cache skips already-judged pairs.

NOTES
  - /api/search returns docs run through rag_controller.clean_rag_references,
    which replaces text/snippet with a placeholder for Trine Day docs (copyright
    protection). The labeler detects this and grades from title+source+subset.
  - The DeepInfra backend posts to the native /v1/inference/{model} endpoint
    and Llama-3-templates the prompt internally.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_DEFAULT_MODEL = "claude-haiku-4-5"

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/inference"
DEEPINFRA_DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"

JUDGE_PROMPT = """You are evaluating whether a document excerpt is relevant to a search query.
The corpus contains declassified U.S. government documents, investigative reporting,
and historical analysis related to political history, intelligence agencies, and assassinations.

QUERY:
{query}

DOCUMENT
Title:   {title}
Source:  {source}
Subset:  {subset}

Excerpt:
{text}

Grade the relevance of this DOCUMENT to the QUERY on a 0–3 scale:
  0 = Not relevant. The document does not address the query topic.
  1 = Marginally relevant. Touches related themes but does not address the query.
  2 = Relevant. The document substantively addresses the query.
  3 = Highly relevant. The document directly answers the query or is a primary source.

Respond with EXACTLY one digit on the first line: 0, 1, 2, or 3.
You may add a brief one-line rationale on line 2."""


# ---------------------------------------------------------------------------
# Judge backends
# ---------------------------------------------------------------------------

def parse_grade(text: str) -> Optional[int]:
    m = re.search(r"\b([0-3])\b", text or "")
    return int(m.group(1)) if m else None


def judge_anthropic(
    api_key: str, model: str, prompt: str, timeout: float = 60.0
) -> Tuple[Optional[int], str]:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": 80,
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post(ANTHROPIC_URL, headers=headers, json=body, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    text = ""
    if isinstance(j.get("content"), list):
        text = "".join(c.get("text", "") for c in j["content"] if c.get("type") == "text")
    return parse_grade(text), text.strip()


def judge_stub(prompt: str) -> Tuple[Optional[int], str]:
    # Deterministic by hash — meaningless grades, useful only for pipeline testing.
    h = sum(ord(c) for c in prompt) % 4
    return h, f"stub_grade={h}"


def _llama3_chat_template(user_prompt: str) -> str:
    """
    Llama-3 / Llama-3.1 instruct chat template. DeepInfra's native
    /v1/inference/{model} endpoint expects a pre-templated `input` string for
    these models — no system prompt; just user → assistant.
    """
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def judge_deepinfra(
    api_key: str, model: str, prompt: str, timeout: float = 60.0
) -> Tuple[Optional[int], str]:
    """
    DeepInfra native inference endpoint. Posts to {DEEPINFRA_BASE_URL}/{model}
    with a Llama-3-templated `input`. Response shape:
        {"results": [{"generated_text": "..."}], "inference_status": {...}, ...}
    """
    url = f"{DEEPINFRA_BASE_URL}/{model}"
    headers = {
        "Authorization": f"bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "input": _llama3_chat_template(prompt),
        "max_new_tokens": 80,
        "temperature": 0.0,
        "stop": ["<|eot_id|>"],
    }
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    text = ""
    if isinstance(j.get("results"), list) and j["results"]:
        text = j["results"][0].get("generated_text", "") or ""
    # Some response variants put it at the top level; fall back gracefully.
    if not text and isinstance(j.get("generated_text"), str):
        text = j["generated_text"]
    return parse_grade(text), text.strip()


# ---------------------------------------------------------------------------
# Retrieval pooling
# ---------------------------------------------------------------------------

def call_search(
    base_url: str, query: str, variant: int, top_k: int, timeout: float = 60.0
) -> List[Dict[str, Any]]:
    payload = {"query": query, "top_k": top_k, "shard_k": 20, "rag_algo_type": variant}
    r = requests.post(f"{base_url.rstrip('/')}/api/search", json=payload, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    if not body.get("ok"):
        raise RuntimeError(body.get("error", "search failed"))
    return body.get("results", []) or []


def doc_id_of(doc: Dict[str, Any], field: str) -> Optional[str]:
    v = doc.get(field)
    return None if v is None else str(v)


def truncate_words(s: str, max_words: int) -> str:
    words = (s or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " …"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

CacheKey = Tuple[str, str]


def load_cache(path: Path) -> Dict[CacheKey, Tuple[Optional[int], str]]:
    out: Dict[CacheKey, Tuple[Optional[int], str]] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            key = (rec["query"], str(rec["doc_id"]))
            grade = rec.get("grade")
            out[key] = (int(grade) if grade is not None else None, rec.get("rationale", ""))
    return out


def append_cache(
    path: Path, query: str, doc_id: str, grade: Optional[int], rationale: str
) -> None:
    with open(path, "a") as f:
        f.write(json.dumps({
            "query": query,
            "doc_id": doc_id,
            "grade": grade,
            "rationale": rationale,
        }) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--queries", required=True, type=Path,
                    help="seed JSONL with one record per line: {\"query\": ...} or {\"query\": ..., \"probe\": true}")
    ap.add_argument("--base-url", default="http://localhost:5000")
    ap.add_argument("--variants", default="1,2,3",
                    help="rag_algo_type values to pool from (default: 1,2,3)")
    ap.add_argument("--pool-top-k", type=int, default=30)
    ap.add_argument("--id-field", default="row_id",
                    help="result field used as the doc identifier (default: row_id)")
    ap.add_argument("--max-text-words", type=int, default=400)
    ap.add_argument("--judge", choices=["anthropic", "deepinfra", "stub"],
                    default="anthropic",
                    help="LLM backend used for relevance grading (default: anthropic)")
    ap.add_argument("--judge-model", default=None,
                    help="judge model name (default: backend-appropriate)")
    ap.add_argument("--judge-stub", action="store_true",
                    help="alias for --judge stub (kept for backward compat)")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--cache", type=Path, default=Path("scripts/eval/judge_cache.jsonl"))
    ap.add_argument("--max-queries", type=int, default=0, help="0 = use all (default)")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds to sleep between judge calls")
    args = ap.parse_args()

    variants = [int(v) for v in args.variants.split(",") if v.strip()]

    # Resolve judge backend (legacy --judge-stub overrides --judge)
    judge_backend = "stub" if args.judge_stub else args.judge

    api_key: Optional[str] = None
    if judge_backend == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("Set ANTHROPIC_API_KEY or use --judge {deepinfra,stub}")
        judge_model = args.judge_model or ANTHROPIC_DEFAULT_MODEL
    elif judge_backend == "deepinfra":
        api_key = os.environ.get("DEEPINFRA_API_KEY")
        if not api_key:
            raise SystemExit("Set DEEPINFRA_API_KEY or use --judge {anthropic,stub}")
        judge_model = args.judge_model or DEEPINFRA_DEFAULT_MODEL
    else:  # stub
        judge_model = args.judge_model or "stub"

    print(f"[labeler] judge backend = {judge_backend}, model = {judge_model}")

    # Load seed queries
    seeds: List[Dict[str, Any]] = []
    with open(args.queries) as f:
        for ln_no, ln in enumerate(f, 1):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            try:
                seeds.append(json.loads(ln))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{args.queries}:{ln_no}: bad JSON: {e}")
    if args.max_queries:
        seeds = seeds[: args.max_queries]
    if not seeds:
        raise SystemExit("No seed queries loaded")

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(args.cache)
    print(f"[labeler] {len(cache)} cached judgments loaded from {args.cache}")

    n_in = sum(1 for s in seeds if not s.get("probe"))
    n_pr = len(seeds) - n_in
    print(f"[labeler] {len(seeds)} queries — in-domain: {n_in}, probe: {n_pr}")

    out_records: List[Dict[str, Any]] = []
    judged_this_run = 0
    cache_hits = 0
    skipped_protected = 0

    for qi, seed in enumerate(seeds):
        q = (seed.get("query") or "").strip()
        if not q:
            continue
        if seed.get("probe"):
            out_records.append({"query": q, "probe": True})
            print(f"  [{qi+1}/{len(seeds)}] PROBE — {q[:64]!r}")
            continue

        # ---- pool unique docs across variants ----
        pool: Dict[str, Dict[str, Any]] = {}
        for v in variants:
            try:
                results = call_search(args.base_url, q, v, args.pool_top_k)
            except Exception as e:
                print(f"  [WARN] q{qi} v{v}: {e}")
                continue
            for r in results:
                d = doc_id_of(r, args.id_field)
                if d and d not in pool:
                    pool[d] = r

        if not pool:
            print(f"  [{qi+1}/{len(seeds)}] {q[:64]!r} — empty candidate pool")
            out_records.append({"query": q, "labels": {}, "id_field": args.id_field})
            continue

        # ---- judge each (q, doc) pair ----
        labels: Dict[str, int] = {}
        for d, doc in pool.items():
            key = (q, d)
            if key in cache:
                grade, _ = cache[key]
                cache_hits += 1
            else:
                title = str(doc.get("title") or "").strip()
                source = str(doc.get("source") or "").strip()
                subset = str(doc.get("subset") or "").strip()
                text = str(doc.get("text") or doc.get("snippet") or "").strip()
                if "protected by copyright" in text.lower():
                    text = "(full text unavailable — grade from title/source/subset)"
                    skipped_protected += 1
                else:
                    text = truncate_words(text, args.max_text_words)
                prompt = JUDGE_PROMPT.format(
                    query=q,
                    title=title or "(none)",
                    source=source or "(none)",
                    subset=subset or "(none)",
                    text=text or "(empty)",
                )
                try:
                    if judge_backend == "stub":
                        grade, rationale = judge_stub(prompt)
                    elif judge_backend == "deepinfra":
                        grade, rationale = judge_deepinfra(api_key, judge_model, prompt)
                    else:
                        grade, rationale = judge_anthropic(api_key, judge_model, prompt)
                except Exception as e:
                    print(f"      [WARN] judge failed for doc={d}: {e}")
                    grade, rationale = None, f"error: {e}"
                append_cache(args.cache, q, d, grade, rationale)
                cache[key] = (grade, rationale)
                judged_this_run += 1
                if args.sleep > 0:
                    time.sleep(args.sleep)
            if grade is not None and grade > 0:
                labels[d] = int(grade)

        out_records.append({"query": q, "labels": labels, "id_field": args.id_field})
        print(
            f"  [{qi+1}/{len(seeds)}] {q[:64]!r} — pool={len(pool)} "
            f"relevant={len(labels)} (cache hits so far={cache_hits}, judged={judged_this_run})"
        )

    # ---- write labels.jsonl ----
    with open(args.output, "w") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")

    print()
    print(f"[labeler] judged this run: {judged_this_run}")
    print(f"[labeler] cache hits:      {cache_hits}")
    print(f"[labeler] protected docs:  {skipped_protected} (graded from metadata only)")
    print(f"[labeler] labels file →    {args.output}")
    print(f"[labeler] cache file →     {args.cache}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[labeler] interrupted.", file=sys.stderr)
        sys.exit(130)
