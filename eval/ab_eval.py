#!/usr/bin/env python3
"""
A/B evaluation harness for hybrid RAG retrieval variants.

Compares the rag_algo_choice variants (1, 2, 3 — see rag_controller.search_references)
against a labeled eval set, reporting paired metrics with bootstrap CIs and
Wilcoxon p-values (Holm-corrected across the family of pairwise comparisons).

Run the Flask app first (so /api/search is up), then point this at it.

USAGE
  python scripts/eval/ab_eval.py \
      --eval-set scripts/eval/labels.jsonl \
      --variants 1,2,3 \
      --top-k 20 \
      --base-url http://localhost:5000 \
      --output-dir scripts/eval/runs

EVAL SET FORMAT (JSONL — one record per line; '#' lines and blank lines ignored)

  Binary labels:
    {"query": "What was on Oswald's bus ticket?", "relevant_doc_ids": ["row_42", "row_99"]}

  Graded labels (preferred for nDCG):
    {"query": "...", "labels": {"row_42": 3, "row_99": 2, "row_7": 1}}

  Optional per-record override of which result key to match against
  (defaults to "row_id" — also supports "title" or "source"):
    {"query": "...", "relevant_doc_ids": ["..."], "id_field": "title"}

OUTPUTS
  output-dir/raw_<run>.jsonl       — one record per (query, variant) with the
                                     ranked id list, latency, and labels.
  output-dir/metrics_<run>.csv     — per-query metrics, one row per (variant, query).
  Console — aggregate table + pairwise comparisons with CI/p/p_holm.

DEPENDENCIES
  Standard library + `requests`.

NOTES
  - /api/search is the retrieval-only endpoint and accepts:
        {"query", "top_k", "shard_k", "rag_algo_type", "subsets"}
    rag_algo_type maps directly to rag_algo_choice in search_references.
  - Variant 0 falls through to _hybrid_search_sqlite (== variant 1) per the
    match statement in search_references — pass 1, 2, 3 to compare implementations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# ---------------------------------------------------------------------------
# Eval set loading
# ---------------------------------------------------------------------------

@dataclass
class Labeled:
    query: str
    labels: Dict[str, int]   # doc_id -> grade (binary uses 1)
    id_field: str = "row_id"
    probe: bool = False      # off-corpus probe — judged on declination, not ranking


def load_eval_set(path: Path) -> List[Labeled]:
    items: List[Labeled] = []
    with open(path) as f:
        for ln_no, ln in enumerate(f, 1):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{ln_no}: bad JSON: {e}")
            q = (rec.get("query") or "").strip()
            if not q:
                raise SystemExit(f"{path}:{ln_no}: missing 'query'")
            id_field = rec.get("id_field", "row_id")
            probe = bool(rec.get("probe", False))
            if probe:
                items.append(Labeled(query=q, labels={}, id_field=id_field, probe=True))
                continue
            if isinstance(rec.get("labels"), dict):
                labels = {str(k): int(v) for k, v in rec["labels"].items()}
            elif "relevant_doc_ids" in rec:
                labels = {str(d): 1 for d in rec["relevant_doc_ids"]}
            else:
                raise SystemExit(
                    f"{path}:{ln_no}: needs 'labels' / 'relevant_doc_ids' / 'probe: true'"
                )
            items.append(Labeled(query=q, labels=labels, id_field=id_field))
    return items


# ---------------------------------------------------------------------------
# Retrieval calls
# ---------------------------------------------------------------------------

def call_search(
    base_url: str,
    query: str,
    variant: int,
    top_k: int,
    shard_k: int = 20,
    subsets: Optional[List[str]] = None,
    timeout: float = 60.0,
) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    """Returns (results, latency_ms, response_body). Body carries v4 gate metadata."""
    payload: Dict[str, Any] = {
        "query": query,
        "top_k": top_k,
        "shard_k": shard_k,
        "rag_algo_type": variant,
    }
    if subsets:
        payload["subsets"] = subsets
    t0 = time.perf_counter()
    r = requests.post(f"{base_url.rstrip('/')}/api/search", json=payload, timeout=timeout)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if r.status_code != 200:
        raise RuntimeError(f"variant={variant} HTTP {r.status_code}: {r.text[:300]}")
    body = r.json()
    if not body.get("ok"):
        raise RuntimeError(f"variant={variant} ok=False: {body.get('error')}")
    return (body.get("results", []) or []), elapsed_ms, body


def get_doc_id(result: Dict[str, Any], id_field: str) -> Optional[str]:
    v = result.get(id_field)
    return None if v is None else str(v)


# ---------------------------------------------------------------------------
# Metrics  (per query — labels: {doc_id -> grade>=0})
# ---------------------------------------------------------------------------

def _relevant(labels: Dict[str, int]) -> set:
    return {d for d, g in labels.items() if g > 0}


def recall_at_k(ranked_ids: List[str], labels: Dict[str, int], k: int) -> float:
    rel = _relevant(labels)
    if not rel:
        return float("nan")
    hit = sum(1 for d in ranked_ids[:k] if d in rel)
    return hit / len(rel)


def precision_at_k(ranked_ids: List[str], labels: Dict[str, int], k: int) -> float:
    if k <= 0:
        return float("nan")
    rel = _relevant(labels)
    if not rel:
        return float("nan")
    hit = sum(1 for d in ranked_ids[:k] if d in rel)
    return hit / k


def mrr(ranked_ids: List[str], labels: Dict[str, int]) -> float:
    rel = _relevant(labels)
    if not rel:
        return float("nan")
    for i, d in enumerate(ranked_ids, 1):
        if d in rel:
            return 1.0 / i
    return 0.0


def average_precision(ranked_ids: List[str], labels: Dict[str, int]) -> float:
    rel = _relevant(labels)
    if not rel:
        return float("nan")
    hits = 0
    sumprec = 0.0
    for i, d in enumerate(ranked_ids, 1):
        if d in rel:
            hits += 1
            sumprec += hits / i
    return sumprec / len(rel)


def _dcg(grades: List[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked_ids: List[str], labels: Dict[str, int], k: int) -> float:
    if not labels:
        return float("nan")
    grades_at_k = [labels.get(d, 0) for d in ranked_ids[:k]]
    ideal = sorted(labels.values(), reverse=True)[:k]
    if not any(ideal):
        return float("nan")
    idcg = _dcg(ideal)
    return _dcg(grades_at_k) / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(
    deltas: List[float],
    iters: int = 10000,
    seed: int = 1234,
    alpha: float = 0.05,
) -> Tuple[float, float, float, float]:
    """
    Returns (observed_mean, ci_lo, ci_hi, p_two_sided).
    Resamples the per-query deltas with replacement; p is the two-sided
    fraction of bootstrap means crossing zero.
    """
    deltas = [d for d in deltas if not math.isnan(d)]
    if not deltas:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(deltas)
    obs_mean = sum(deltas) / n
    means: List[float] = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += deltas[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(iters * alpha / 2)]
    hi = means[min(iters - 1, int(iters * (1 - alpha / 2)))]
    above = sum(1 for m in means if m >= 0)
    below = sum(1 for m in means if m <= 0)
    p = 2.0 * min(above, below) / iters
    return obs_mean, lo, hi, min(p, 1.0)


def wilcoxon_signed_rank(deltas: List[float]) -> Tuple[float, float]:
    """
    Two-sided Wilcoxon signed-rank with normal approximation and tie-averaged
    ranks. Returns (W, p). NaN if n<6 after dropping zeros/NaN.
    """
    nz = [d for d in deltas if d != 0 and not math.isnan(d)]
    n = len(nz)
    if n < 6:
        return float("nan"), float("nan")
    abs_pairs = sorted(((abs(d), d) for d in nz), key=lambda x: x[0])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_pairs[j + 1][0] == abs_pairs[i][0]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    w_plus = sum(r for r, (_, d) in zip(ranks, abs_pairs) if d > 0)
    w_minus = sum(r for r, (_, d) in zip(ranks, abs_pairs) if d < 0)
    W = min(w_plus, w_minus)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return W, float("nan")
    z = (W - mu) / sigma
    # two-sided p via standard normal
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return W, p


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni step-down adjustment. Preserves input order."""
    indexed = [(p, i) for i, p in enumerate(p_values) if not math.isnan(p)]
    indexed.sort(key=lambda x: x[0])
    m = len(indexed)
    adj = [float("nan")] * len(p_values)
    running = 0.0
    for k, (p, orig_idx) in enumerate(indexed):
        a = min(1.0, (m - k) * p)
        running = max(running, a)
        adj[orig_idx] = running
    return adj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--eval-set", required=True, type=Path)
    ap.add_argument("--variants", default="1,2,3",
                    help="comma-separated rag_algo_type values (default: 1,2,3)")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--shard-k", type=int, default=20)
    ap.add_argument("--metric-ks", default="5,10",
                    help="comma-separated ks for @k metrics (default: 5,10)")
    ap.add_argument("--base-url", default="http://localhost:5000")
    ap.add_argument("--output-dir", type=Path, default=Path("scripts/eval/runs"))
    ap.add_argument("--bootstrap-iters", type=int, default=10000)
    ap.add_argument("--max-queries", type=int, default=0,
                    help="0 = use all (default)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-server", action="store_true",
                    help="skip retrieval; recompute metrics from an existing raw_*.jsonl in --output-dir")
    ap.add_argument("--raw-file", type=Path, default=None,
                    help="when used with --no-server, a specific raw_*.jsonl to reuse")
    args = ap.parse_args()

    variants = [int(v) for v in args.variants.split(",") if v.strip()]
    metric_ks = sorted({int(k) for k in args.metric_ks.split(",") if k.strip()})
    if not variants:
        raise SystemExit("--variants must list at least one value")
    if not metric_ks:
        raise SystemExit("--metric-ks must list at least one value")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    raw_path = args.output_dir / f"raw_{run_id}.jsonl"
    metrics_path = args.output_dir / f"metrics_{run_id}.csv"

    eval_set = load_eval_set(args.eval_set)
    if args.max_queries:
        eval_set = eval_set[: args.max_queries]
    if not eval_set:
        raise SystemExit("No eval items loaded")

    # ---- run retrieval (or reload) ----
    raw: List[Dict[str, Any]] = []
    if args.no_server:
        src = args.raw_file or sorted(args.output_dir.glob("raw_*.jsonl"))[-1]
        print(f"[ab_eval] reusing {src}")
        with open(src) as f:
            for ln in f:
                if ln.strip():
                    raw.append(json.loads(ln))
        # honor --variants filter on reused data
        raw = [r for r in raw if int(r["variant"]) in variants]
        if not raw:
            raise SystemExit("Reused raw file has no records matching --variants")
    else:
        print(f"[ab_eval] {len(eval_set)} queries × {len(variants)} variants → {raw_path}")
        with open(raw_path, "w") as f:
            for qi, item in enumerate(eval_set):
                for variant in variants:
                    body: Dict[str, Any] = {}
                    try:
                        results, ms, body = call_search(
                            args.base_url, item.query, variant,
                            top_k=args.top_k, shard_k=args.shard_k,
                        )
                    except Exception as e:
                        print(f"  [WARN] q{qi} v{variant}: {e}")
                        results, ms = [], -1.0
                    ranked = [get_doc_id(r, item.id_field) for r in results]
                    ranked = [d for d in ranked if d is not None]
                    # Top-1 BM25: prefer the body-level field (v4 reports the
                    # PRE-gate score even when results are declined to []) and
                    # fall back to results[0] for v1/v2/v3.
                    top1: float = float("nan")
                    body_top1 = body.get("top1_score")
                    if body_top1 is not None:
                        try:
                            top1 = float(body_top1)
                        except (TypeError, ValueError):
                            top1 = float("nan")
                    elif results:
                        v = results[0].get("score_bm25")
                        if v is None:
                            v = results[0].get("score")
                        try:
                            if v is not None:
                                top1 = float(v)
                        except (TypeError, ValueError):
                            top1 = float("nan")
                    rec = {
                        "qi": qi,
                        "variant": variant,
                        "query": item.query,
                        "ranked": ranked,
                        "latency_ms": ms,
                        "n_results": len(ranked),
                        "top1_score": top1,
                        "labels": item.labels,
                        "id_field": item.id_field,
                        "probe": item.probe,
                        # v4 gate metadata (None / absent for v1/v2/v3)
                        "gate_decision": body.get("gate_decision"),
                        "gate_reason": body.get("gate_reason"),
                        "n_canonical_entities": body.get("n_canonical_entities"),
                        "n_non_location_entities": body.get("n_non_location_entities"),
                        "fts_branch_used": body.get("fts_branch_used"),
                        "pre_gate_n_results": body.get("pre_gate_n_results"),
                    }
                    raw.append(rec)
                    f.write(json.dumps(rec) + "\n")
                tag = "PROBE " if item.probe else ""
                print(f"  [{qi+1}/{len(eval_set)}] {tag}{item.query[:64]!r}")

    # ---- per-query metrics ----
    metric_names = ["mrr", "map"] + [
        f"{base}@{k}" for k in metric_ks for base in ("recall", "precision", "ndcg")
    ]
    # In-domain queries (have labels) → ranking metrics
    per_q: Dict[int, List[Dict[str, float]]] = {v: [] for v in variants}
    # Probe queries → declination metrics (n_results, no_result_rate, top1_score)
    per_q_probe: Dict[int, List[Dict[str, float]]] = {v: [] for v in variants}
    for r in raw:
        v = int(r["variant"])
        ids = r["ranked"]
        is_probe = bool(r.get("probe", False))
        if is_probe:
            per_q_probe[v].append({
                "qi": r["qi"],
                "latency_ms": r["latency_ms"],
                "n_results": float(r["n_results"]),
                "no_result": 1.0 if r["n_results"] == 0 else 0.0,
                "top1_score": float(r.get("top1_score", float("nan"))),
            })
            continue
        labels = {str(k): int(v_) for k, v_ in r["labels"].items()}
        m: Dict[str, float] = {
            "qi": r["qi"],
            "latency_ms": r["latency_ms"],
            "no_result": 1.0 if len(ids) == 0 else 0.0,
            "mrr": mrr(ids, labels),
            "map": average_precision(ids, labels),
        }
        for k in metric_ks:
            m[f"recall@{k}"] = recall_at_k(ids, labels, k)
            m[f"precision@{k}"] = precision_at_k(ids, labels, k)
            m[f"ndcg@{k}"] = ndcg_at_k(ids, labels, k)
        per_q[v].append(m)

    # ---- aggregate table: in-domain queries (ranking metrics) ----
    n_in = len(per_q[variants[0]]) if variants else 0
    n_pr = len(per_q_probe[variants[0]]) if variants else 0
    print()
    print(f"=== In-domain queries (n={n_in}) — higher is better ===")
    print("=" * (18 + 14 * len(variants)))
    print(f"{'metric':<18}" + "".join(f"{f'v{v}':>14}" for v in variants))
    print("-" * (18 + 14 * len(variants)))
    for name in metric_names:
        line = f"{name:<18}"
        for v in variants:
            vals = [m[name] for m in per_q[v] if not math.isnan(m[name])]
            mean = sum(vals) / len(vals) if vals else float("nan")
            line += f"{mean:>14.4f}"
        print(line)
    # latency p50/p95
    line = f"{'latency_ms p50/p95':<18}"
    for v in variants:
        lats = sorted(m["latency_ms"] for m in per_q[v] if m["latency_ms"] >= 0)
        if lats:
            p50 = lats[len(lats) // 2]
            p95 = lats[min(len(lats) - 1, int(len(lats) * 0.95))]
            line += f" {p50:>6.0f}/{p95:<6.0f}"
        else:
            line += f"{'n/a':>14}"
    print(line)
    # no-result rate
    line = f"{'no_result_rate':<18}"
    for v in variants:
        vals = [m["no_result"] for m in per_q[v]]
        rate = sum(vals) / len(vals) if vals else float("nan")
        line += f"{rate:>14.4f}"
    print(line)
    print("=" * (18 + 14 * len(variants)))

    # ---- aggregate table: probe queries (declination metrics) ----
    if n_pr > 0:
        print()
        print(f"=== Off-corpus probe queries (n={n_pr}) — for declination, lower n_results / lower top1 / higher no_result_rate is better ===")
        print("=" * (22 + 14 * len(variants)))
        print(f"{'probe metric':<22}" + "".join(f"{f'v{v}':>14}" for v in variants))
        print("-" * (22 + 14 * len(variants)))
        for name, label in [
            ("no_result", "no_result_rate ↑"),
            ("n_results", "mean_n_results ↓"),
            ("top1_score", "mean_top1_score ↓"),
        ]:
            line = f"{label:<22}"
            for v in variants:
                vals = [m[name] for m in per_q_probe[v] if not math.isnan(m[name])]
                mean = sum(vals) / len(vals) if vals else float("nan")
                line += f"{mean:>14.4f}"
            print(line)
        print("=" * (22 + 14 * len(variants)))
        print("(Sample size on probes is too small for paired stats — interpret as directional only.)")

    # ---- pairwise comparisons ----
    pairs = [(variants[i], variants[j]) for i in range(len(variants)) for j in range(i + 1, len(variants))]
    if not pairs:
        print("\n(only one variant — no pairwise comparisons.)")
    else:
        rows = []
        for a, b in pairs:
            qis = sorted(set(m["qi"] for m in per_q[a]) & set(m["qi"] for m in per_q[b]))
            ma = {m["qi"]: m for m in per_q[a]}
            mb = {m["qi"]: m for m in per_q[b]}
            for name in metric_names:
                deltas = []
                for qi in qis:
                    da, db = ma[qi][name], mb[qi][name]
                    if not (math.isnan(da) or math.isnan(db)):
                        deltas.append(db - da)
                if not deltas:
                    continue
                mean, lo, hi, p_boot = paired_bootstrap_ci(
                    deltas, iters=args.bootstrap_iters, seed=args.seed
                )
                _, p_wil = wilcoxon_signed_rank(deltas)
                rows.append((a, b, name, mean, lo, hi, p_boot, p_wil, len(deltas)))
        # Holm correction across the whole family of Wilcoxon p's
        adj = holm_bonferroni([r[7] for r in rows])
        print("\nPairwise (B − A) — paired bootstrap 95% CI, Wilcoxon p, Holm-adjusted p:\n")
        header = (f"{'A':>3} {'B':>3} {'metric':<14} {'B-A':>10} "
                  f"{'95% CI':>22} {'p_boot':>9} {'p_wilc':>9} {'p_holm':>9}  n")
        print(header)
        print("-" * len(header))
        for (a, b, name, mean, lo, hi, p_boot, p_wil, n), p_holm in zip(rows, adj):
            ci = f"[{lo:+.4f},{hi:+.4f}]"
            sig = "*" if (not math.isnan(p_holm) and p_holm < 0.05) else " "
            ph = f"{p_holm:>9.4f}" if not math.isnan(p_holm) else f"{'n/a':>9}"
            pw = f"{p_wil:>9.4f}" if not math.isnan(p_wil) else f"{'n/a':>9}"
            print(f"{a:>3} {b:>3} {name:<14} {mean:>+10.4f} {ci:>22} "
                  f"{p_boot:>9.4f} {pw} {ph}{sig}  {n}")
        print("\n  * = Holm-adjusted p < 0.05")

    # ---- per-query CSV (in-domain + probe rows; probe rows leave ranking metrics blank) ----
    fieldnames = ["variant", "qi", "probe", "query", "n_results", "latency_ms",
                  "no_result", "top1_score"] + metric_names
    # Index raw records by (variant, qi) for safe pairing
    raw_idx: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for r in raw:
        raw_idx[(int(r["variant"]), int(r["qi"]))] = r
    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for v in variants:
            for m in per_q[v]:
                r = raw_idx[(v, int(m["qi"]))]
                row = {
                    "variant": v,
                    "qi": int(m["qi"]),
                    "probe": 0,
                    "query": r["query"],
                    "n_results": r["n_results"],
                    "latency_ms": m["latency_ms"],
                    "no_result": m["no_result"],
                    "top1_score": r.get("top1_score", ""),
                }
                for name in metric_names:
                    row[name] = m[name]
                w.writerow(row)
            for m in per_q_probe[v]:
                r = raw_idx[(v, int(m["qi"]))]
                row = {
                    "variant": v,
                    "qi": int(m["qi"]),
                    "probe": 1,
                    "query": r["query"],
                    "n_results": r["n_results"],
                    "latency_ms": m["latency_ms"],
                    "no_result": m["no_result"],
                    "top1_score": m["top1_score"],
                }
                for name in metric_names:
                    row[name] = ""
                w.writerow(row)

    if not args.no_server:
        print(f"\n[ab_eval] Raw results       → {raw_path}")
    print(f"[ab_eval] Per-query metrics → {metrics_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ab_eval] interrupted.", file=sys.stderr)
        sys.exit(130)
