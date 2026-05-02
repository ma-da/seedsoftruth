#!/usr/bin/env python3
"""
Calibrate a BM25 top-1 score floor for v4's min-gate, using raw output from a
prior ab_eval run. Per variant, sweeps thresholds and reports:

  TD  true-decline rate    — probe queries blocked at this threshold (higher = better)
  FD  false-decline rate   — in-domain queries blocked at this threshold (lower = better)
  J   Youden's J = TD − FD — overall separation quality (higher = better)

Picks the J-maximizing threshold per variant and prints the full per-query
score distributions so you can see where probe and in-domain queries overlap.

USAGE
  python eval/calibrate_floor.py --raw eval/runs/raw_<ts>.jsonl
  python eval/calibrate_floor.py --raw eval/runs/raw_<ts>.jsonl --variants 2 \
      --threshold-min 10 --threshold-max 50 --threshold-step 0.5

A query is "declined" by the gate iff:
  - n_results == 0, OR
  - top-1 BM25 score is below the threshold (or NaN).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def fmt_top1(t: Any) -> str:
    if t is None:
        return "  --  "
    try:
        x = float(t)
        if math.isnan(x):
            return "  --  "
        return f"{x:6.2f}"
    except (TypeError, ValueError):
        return "  --  "


def is_declined(top1: Any, n_results: int, threshold: float) -> bool:
    if n_results == 0:
        return True
    if top1 is None:
        return True
    try:
        x = float(top1)
        if math.isnan(x):
            return True
        return x < threshold
    except (TypeError, ValueError):
        return True


def sort_key(r: Dict[str, Any]) -> float:
    """Order rows by top1 score, with NaN/missing first."""
    t = r.get("top1_score")
    if t is None:
        return float("-inf")
    try:
        x = float(t)
        if math.isnan(x):
            return float("-inf")
        return x
    except (TypeError, ValueError):
        return float("-inf")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw", required=True, type=Path,
                    help="path to raw_*.jsonl from a prior ab_eval run")
    ap.add_argument("--variants", default="1,2,3")
    ap.add_argument("--threshold-min", type=float, default=0.0)
    ap.add_argument("--threshold-max", type=float, default=50.0)
    ap.add_argument("--threshold-step", type=float, default=1.0)
    args = ap.parse_args()

    rows = [json.loads(ln) for ln in open(args.raw) if ln.strip()]
    variants = [int(v) for v in args.variants.split(",") if v.strip()]

    by_v: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_v[int(r["variant"])].append(r)

    print(f"Loaded {len(rows)} records from {args.raw}")

    for v in variants:
        rows_v = by_v.get(v, [])
        in_domain = [r for r in rows_v if not r.get("probe")]
        probes = [r for r in rows_v if r.get("probe")]
        if not in_domain or not probes:
            print(f"\n=== Variant {v} — skipped (need both in-domain and probe rows) ===")
            continue

        print(f"\n{'='*78}")
        print(f"=== Variant {v}  (in-domain n={len(in_domain)}, probe n={len(probes)}) ===")
        print(f"{'='*78}")

        print("\nIn-domain queries (ascending top-1 score — left side declined first):")
        for r in sorted(in_domain, key=sort_key):
            print(f"  qi={r['qi']:>2}  top1={fmt_top1(r.get('top1_score'))}  "
                  f"n={r['n_results']:>2}  {r['query'][:64]}")

        print("\nProbe queries (ascending top-1 score):")
        for r in sorted(probes, key=sort_key):
            print(f"  qi={r['qi']:>2}  top1={fmt_top1(r.get('top1_score'))}  "
                  f"n={r['n_results']:>2}  {r['query']}")

        # ---- threshold sweep ----
        print(f"\nThreshold sweep:")
        print(f"  {'thresh':>7}  {'TD':>5}  {'FD':>5}  {'J':>6}  notes")
        best_j: float = -2.0
        best_t: Optional[float] = None
        best_td: float = 0.0
        best_fd: float = 0.0
        t = args.threshold_min
        while t <= args.threshold_max + 1e-9:
            td = sum(is_declined(r.get("top1_score"), r["n_results"], t) for r in probes) / len(probes)
            fd = sum(is_declined(r.get("top1_score"), r["n_results"], t) for r in in_domain) / len(in_domain)
            j = td - fd
            note = ""
            if t < 1e-6:
                note = "(no gate — baseline)"
            if td == 1.0 and fd == 0.0 and not note:
                note = "PERFECT separation"
            if j > best_j:
                best_j, best_t, best_td, best_fd = j, t, td, fd
            print(f"  {t:7.1f}  {td:5.2f}  {fd:5.2f}  {j:+6.2f}  {note}")
            t += args.threshold_step

        print(f"\n  → Variant {v} J-max: threshold={best_t:.1f}  "
              f"(TD={best_td:.2f}, FD={best_fd:.2f}, J={best_j:+.2f})")
        # Show which queries decline at the optimum
        if best_t is not None:
            decl_probes = [r for r in probes if is_declined(r.get("top1_score"), r["n_results"], best_t)]
            decl_indom  = [r for r in in_domain if is_declined(r.get("top1_score"), r["n_results"], best_t)]
            slip_probes = [r for r in probes if not is_declined(r.get("top1_score"), r["n_results"], best_t)]
            print(f"  At t={best_t:.1f}: declines {len(decl_probes)}/{len(probes)} probes, "
                  f"falsely declines {len(decl_indom)}/{len(in_domain)} in-domain.")
            if slip_probes:
                print(f"  Probe queries that SLIP THROUGH this gate:")
                for r in slip_probes:
                    print(f"    top1={fmt_top1(r.get('top1_score'))}  {r['query']}")
            if decl_indom:
                print(f"  In-domain queries FALSELY declined:")
                for r in decl_indom:
                    print(f"    top1={fmt_top1(r.get('top1_score'))}  {r['query'][:64]}")


if __name__ == "__main__":
    main()
