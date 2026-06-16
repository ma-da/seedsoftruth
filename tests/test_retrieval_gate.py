"""
Tests for retrieval gate + V5 topic boost.

Covers the bug reported on 2026-05-18 where a query whose verbatim text
appears in exactly one chunk (PEERS Substack geoengineering article)
was declined by the V4/V5 min-gate because SQLite FTS5's bm25() scores
phrase-branch matches systematically lower than strict_AND-branch
matches on the same chunk.

Three layers of coverage:

  1. Pure unit tests on rag_controller._min_gate (no DB).
  2. Unit tests on rag_controller._detect_fts_branch (tiny ephemeral DB).
  3. End-to-end V5 retrieval against an in-memory corpus that reproduces
     the production failure mode.

Run with pytest:
    pytest tests/test_retrieval_gate.py -v

Or run as a plain script (no pytest needed):
    python3 tests/test_retrieval_gate.py
"""

from __future__ import annotations

import asyncio
import math
import os
import sqlite3
import sys
import tempfile
import threading
from pathlib import Path
from typing import Iterator, Optional, Tuple

# Make sibling-module imports work when this file is invoked directly,
# via pytest, or from a CI runner.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
# Web-server modules moved under chat_server/ but import each other flat.
for p in (_ROOT / "chat_server", _ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# rag_controller transitively boots a lot of infrastructure (huggingface
# adapters, etc.) at import time, so set a harmless DB path BEFORE we
# import it. The tests build their own DBs and pass them to a custom
# RetrievalState — they never touch HYBRID_DB_PATH after boot.
_TMP_DB_FOR_IMPORT = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
os.environ.setdefault("HYBRID_DB_PATH", _TMP_DB_FOR_IMPORT)

import rag_controller  # noqa: E402
from rag_controller import (  # noqa: E402
    MIN_GATE_SCORE_FLOOR,
    RetrievalState,
    _detect_fts_branch,
    _hybrid_search_sqlite5,
    _min_gate,
)


# --------------------------------------------------------------------------- #
# Test fixtures
# --------------------------------------------------------------------------- #

# Mirror of corpus_to_hybrid_db.SCHEMA_SQL. Inlined so the tests don't depend
# on that script's stability — if its schema changes, the test must change
# too and the diff makes it obvious.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    lookup_id      INTEGER PRIMARY KEY,
    chunk_id       TEXT,
    title          TEXT,
    subset_name    TEXT,
    domain         TEXT,
    source_url     TEXT,
    entities_text  TEXT,
    fulltext_text  TEXT
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT NOT NULL,
    type           TEXT NOT NULL,
    UNIQUE(canonical_name, type)
);

CREATE TABLE IF NOT EXISTS chunk_entities (
    chunk_lookup_id INTEGER NOT NULL,
    entity_id       INTEGER NOT NULL,
    PRIMARY KEY (chunk_lookup_id, entity_id)
);

CREATE TABLE IF NOT EXISTS topics (
    topic_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    domain     TEXT NOT NULL,
    topic_name TEXT NOT NULL,
    UNIQUE(domain, topic_name)
);

CREATE TABLE IF NOT EXISTS chunk_topics (
    chunk_lookup_id INTEGER NOT NULL,
    topic_id        INTEGER NOT NULL,
    PRIMARY KEY (chunk_lookup_id, topic_id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
    entities_text, tokenize='unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS fulltext_fts USING fts5(
    title, fulltext_text, tokenize='unicode61'
);
"""


def _make_test_db() -> Tuple[Path, sqlite3.Connection]:
    """Create a tempfile SQLite DB with the production schema."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return Path(path), conn


def _insert_chunk(
    conn: sqlite3.Connection,
    lookup_id: int,
    title: str,
    text: str,
    subset: str = "TestSet",
    entities: str = "",
) -> None:
    """Insert a chunk + matching FTS rows. FTS rowid == lookup_id by convention."""
    conn.execute(
        "INSERT INTO chunks(lookup_id, chunk_id, title, subset_name, domain, "
        "source_url, entities_text, fulltext_text) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (lookup_id, f"test-chunk-{lookup_id}", title, subset, "test.example",
         f"https://test.example/{lookup_id}", entities, text),
    )
    conn.execute(
        "INSERT INTO fulltext_fts(rowid, title, fulltext_text) VALUES (?, ?, ?)",
        (lookup_id, title, text),
    )
    conn.execute(
        "INSERT INTO entities_fts(rowid, entities_text) VALUES (?, ?)",
        (lookup_id, entities or ""),
    )
    conn.commit()


def _make_test_state(db_path: Path) -> RetrievalState:
    """Build a RetrievalState wired to a test DB. Loads spaCy once per
    test process — slow on first call, free thereafter."""
    import spacy
    nlp = spacy.load("en_core_web_sm")
    return RetrievalState(
        backend="sqlite_hybrid",
        sqlite_db_path=db_path,
        flat_lookup={},
        nlp=nlp,
        _thread_local=threading.local(),
    )


# --------------------------------------------------------------------------- #
# Layer 1: _min_gate unit tests (no DB)
# --------------------------------------------------------------------------- #

def test_min_gate_phrase_match_below_floor_passes():
    """REGRESSION: a phrase-branch match with BM25 below the floor must
    still pass. The user's geoengineering query produced top1=11.32 on
    the phrase branch (floor=17.0); before the fix this declined and
    returned 0 results."""
    passed, reason = _min_gate(
        entity_terms=[],
        fts_branch="phrase",
        top1_score=11.32,
        score_floor=17.0,
    )
    assert passed is True
    assert reason == "pass_phrase_match", f"expected pass_phrase_match, got {reason}"


def test_min_gate_phrase_match_above_floor_passes():
    """Phrase branch with a high score also passes — no double-counting."""
    passed, reason = _min_gate(
        entity_terms=["foo"],
        fts_branch="phrase",
        top1_score=42.0,
        score_floor=17.0,
    )
    assert passed is True
    assert reason == "pass_phrase_match"


def test_min_gate_phrase_match_ignores_no_entities_rule():
    """The 'no entities + broad_or' guard does NOT apply to phrase
    matches — a verbatim phrase match is never incidental."""
    passed, reason = _min_gate(
        entity_terms=[],          # no canonical entities
        fts_branch="phrase",
        top1_score=8.0,           # also below floor
        score_floor=17.0,
    )
    assert passed is True
    assert reason == "pass_phrase_match"


def test_min_gate_strict_and_above_floor_passes():
    """A strict_AND match with entities and a score above the floor passes
    with reason "pass"."""
    passed, reason = _min_gate(
        entity_terms=["epstein"],
        fts_branch="strict_and",
        top1_score=20.0,
        score_floor=17.0,
    )
    assert passed is True
    assert reason == "pass"


def test_min_gate_strict_and_below_floor_declines():
    """Strict_AND retrieval still gets gated on score floor — this is
    the intended behavior for off-corpus probes that happen to have a
    few matching tokens but no real match."""
    passed, reason = _min_gate(
        entity_terms=[],
        fts_branch="strict_and",
        top1_score=10.0,
        score_floor=17.0,
    )
    assert passed is False
    assert reason == "score_below_floor"


def test_min_gate_broad_or_no_entities_declines():
    """The Switzerland-style guard: broad_OR match with no canonical
    entities means the match is incidental."""
    passed, reason = _min_gate(
        entity_terms=[],
        fts_branch="broad_or",
        top1_score=20.0,           # above floor, but...
        score_floor=17.0,
    )
    assert passed is False
    assert reason == "no_entities_broad_or_only"


def test_min_gate_broad_or_with_entities_passes():
    """A broad_OR match above the floor passes when canonical entities are
    present, with reason "pass"."""
    passed, reason = _min_gate(
        entity_terms=["switzerland"],
        fts_branch="broad_or",
        top1_score=20.0,
        score_floor=17.0,
    )
    assert passed is True
    assert reason == "pass"


def test_min_gate_nan_top1_returns_no_results():
    """A NaN top1 score declines the gate with reason "no_results"."""
    passed, reason = _min_gate(
        entity_terms=["foo"],
        fts_branch="strict_and",
        top1_score=float("nan"),
        score_floor=17.0,
    )
    assert passed is False
    assert reason == "no_results"


def test_min_gate_none_top1_returns_no_results():
    """A None top1 score declines the gate with reason "no_results"."""
    passed, reason = _min_gate(
        entity_terms=["foo"],
        fts_branch="strict_and",
        top1_score=None,           # type: ignore[arg-type]
        score_floor=17.0,
    )
    assert passed is False
    assert reason == "no_results"


def test_min_gate_at_floor_exact_passes():
    """Exact-floor scores should pass — the comparison is strict `<`."""
    passed, reason = _min_gate(
        entity_terms=["foo"],
        fts_branch="strict_and",
        top1_score=17.0,
        score_floor=17.0,
    )
    assert passed is True
    assert reason == "pass"


# --------------------------------------------------------------------------- #
# Layer 2: _detect_fts_branch tests (small ephemeral DB)
# --------------------------------------------------------------------------- #

def _branch_test_state():
    """A test DB seeded with three docs, each only matchable on one
    branch, so we can probe _detect_fts_branch deterministically."""
    db_path, conn = _make_test_db()
    # Chunk 1: phrase match candidate — has the exact phrase "the quick brown fox"
    _insert_chunk(conn, 1,
                  title="phrase doc",
                  text="the quick brown fox jumps over the lazy dog and rests.")
    # Chunk 2: strict_AND match candidate — has all of {alpha, beta, gamma}
    # but not as a phrase
    _insert_chunk(conn, 2,
                  title="strict and doc",
                  text="alpha is followed by something. then beta arrives. eventually gamma.")
    # Chunk 3: only contains 'delta', so broad_OR with {delta, foo, bar} matches
    _insert_chunk(conn, 3,
                  title="lone delta",
                  text="delta is alone in this document.")
    state = _make_test_state(db_path)
    return state, db_path


def test_detect_fts_branch_phrase_hit():
    """_detect_fts_branch returns "phrase" when the query's verbatim text
    matches a chunk as an exact phrase."""
    state, _ = _branch_test_state()
    # A 5+-word query whose verbatim text exists in chunk 1.
    branch = _detect_fts_branch(state, "the quick brown fox jumps")
    assert branch == "phrase", f"expected phrase, got {branch!r}"


def test_detect_fts_branch_strict_and_hit():
    """_detect_fts_branch returns "strict_and" when a chunk contains all
    query tokens but not as a contiguous phrase."""
    state, _ = _branch_test_state()
    # No chunk has "alpha beta gamma" as a phrase, but chunk 2 has all three.
    branch = _detect_fts_branch(state, "alpha beta gamma")
    assert branch == "strict_and", f"expected strict_and, got {branch!r}"


def test_detect_fts_branch_broad_or_only_hit():
    """_detect_fts_branch returns "broad_or" when only some query tokens
    match any chunk (no phrase or strict_AND hit)."""
    state, _ = _branch_test_state()
    # No chunk has both 'delta' AND 'foo' AND 'bar', but chunk 3 has 'delta'.
    branch = _detect_fts_branch(state, "delta foo bar")
    assert branch == "broad_or", f"expected broad_or, got {branch!r}"


def test_detect_fts_branch_no_hit():
    """_detect_fts_branch returns "none" when no query token matches any
    chunk."""
    state, _ = _branch_test_state()
    branch = _detect_fts_branch(state, "xyzzy plugh frobnitz")
    assert branch == "none", f"expected none, got {branch!r}"


# --------------------------------------------------------------------------- #
# Layer 3: V5 retrieval integration (reproduces the production bug)
# --------------------------------------------------------------------------- #

def _build_geoengineering_db() -> Tuple[Path, sqlite3.Connection]:
    """Build a DB mirroring the production failure:
    - chunk A: PEERS-Substack-style chunk containing the verbatim 7-word
      phrase the user queried
    - chunk B: WTK-style chunk that mentions 'weather modification' but
      NOT the verbatim phrase
    """
    db_path, conn = _make_test_db()
    _insert_chunk(
        conn, 100,
        title="how-geoengineering-went-from-top",
        text=("In the wake of recent extreme weather events, "
              "public concerns about geoengineering and weather modification "
              "is growing. Yet instead of engaging these concerns in good "
              "faith, official narratives have branded this line of inquiry "
              "as a conspiracy theory."),
        subset="PEERS Substack",
    )
    _insert_chunk(
        conn, 200,
        title="climate-science-rebrand",
        text=("WTK stops Rebranded as Climate Science. There's a Better "
              "Way To Heal the Earth. We present credible evidence and "
              "current information showing that weather modification "
              "technologies are not only real, but that they are being "
              "secretly propagated by multiple actors."),
        subset="WantToKnow.info",
    )
    return db_path, conn


def test_v5_finds_phrase_match_chunk():
    """REGRESSION: V5 must return the chunk whose verbatim text matches
    the user's query, even though the phrase-branch BM25 is below the
    historic score floor."""
    db_path, _ = _build_geoengineering_db()
    state = _make_test_state(db_path)
    out = asyncio.run(rag_controller.search_references(
        state,
        "public concerns about geoengineering and weather modification",
        rag_algo_choice=5,
        top_k=5,
    ))
    assert out["num_results"] >= 1, (
        f"expected ≥1 result, got 0 (gate={out.get('gate_decision')}/"
        f"{out.get('gate_reason')}, top1={out.get('top1_score')})"
    )
    titles = [r.get("title") for r in out["results"]]
    assert any("geoengineering" in (t or "") for t in titles), (
        f"expected the geoengineering chunk in top results, got titles={titles}"
    )
    assert out.get("gate_reason") == "pass_phrase_match"
    assert out.get("fts_branch_used") == "phrase"


def test_v5_phrase_match_wins_over_partial_match():
    """The phrase-matching chunk should rank ahead of a non-phrase
    chunk that shares some tokens but doesn't contain the exact phrase."""
    db_path, _ = _build_geoengineering_db()
    state = _make_test_state(db_path)
    out = asyncio.run(rag_controller.search_references(
        state,
        "public concerns about geoengineering and weather modification",
        rag_algo_choice=5,
        top_k=5,
    ))
    assert out["num_results"] >= 1
    top_subset = out["results"][0].get("subset")
    assert top_subset == "PEERS Substack", (
        f"expected PEERS Substack on top, got {top_subset!r}"
    )


def test_v5_off_corpus_probe_still_declines():
    """REGRESSION: the phrase-bypass must not weaken probe rejection.
    Random word-salad queries never phrase-match, so the score floor
    still applies on broad_or / strict_and branches."""
    db_path, _ = _build_geoengineering_db()
    state = _make_test_state(db_path)
    out = asyncio.run(rag_controller.search_references(
        state,
        "cadmium schooner swallow",
        rag_algo_choice=5,
        top_k=5,
    ))
    assert out["num_results"] == 0, (
        f"expected 0 results for off-corpus probe, got {out['num_results']}"
    )
    assert out.get("gate_decision") == "decline"
    # The exact reason depends on whether any single token matched anywhere;
    # if so it'll be score_below_floor, otherwise no_results / similar.
    assert out.get("gate_reason") != "pass_phrase_match"


def test_v5_works_when_chunk_topics_empty():
    """If chunk_topics is empty (e.g., PEERS Substack ingested without
    --with-topics), V5 must still retrieve correctly — degrading to
    V4-equivalent ranking without the boost."""
    db_path, _ = _build_geoengineering_db()
    state = _make_test_state(db_path)
    out = asyncio.run(rag_controller.search_references(
        state,
        "public concerns about geoengineering and weather modification",
        rag_algo_choice=5,
        top_k=5,
    ))
    # Should still return the phrase-matching chunk
    assert out["num_results"] >= 1
    r0 = out["results"][0]
    # Boost should be the neutral 1.0 since chunks have no topic rows
    assert r0.get("topic_boost") == 1.0, (
        f"expected boost=1.0 with empty chunk_topics, got {r0.get('topic_boost')}"
    )


def test_v5_default_when_no_algo_specified():
    """The new production default: search_references() with no
    rag_algo_choice arg must dispatch to V5 (V5-only fields present)."""
    db_path, _ = _build_geoengineering_db()
    state = _make_test_state(db_path)
    out = asyncio.run(rag_controller.search_references(
        state,
        "public concerns about geoengineering and weather modification",
        top_k=5,
    ))
    # V5-only gate metadata keys
    for k in ("topic_inference_source", "n_query_topics", "boost_alpha"):
        assert k in out, f"missing V5 metadata key {k!r}; got keys={sorted(out)}"


# --------------------------------------------------------------------------- #
# Manual runner — `python3 tests/test_retrieval_gate.py`
# --------------------------------------------------------------------------- #

def _collect_tests() -> Iterator[Tuple[str, callable]]:
    """Test-runner helper; yield (name, callable) pairs for every module-level
    ``test_*`` function, sorted by name."""
    g = globals()
    for name in sorted(g):
        if name.startswith("test_") and callable(g[name]):
            yield name, g[name]


def _run_all_manual() -> int:
    """Test-runner helper; run every collected test without pytest, print a
    PASS/FAIL/ERR line per test plus a summary, and return 0 if all passed
    else 1."""
    passed = []
    failed = []
    for name, fn in _collect_tests():
        try:
            fn()
        except AssertionError as e:
            failed.append((name, f"AssertionError: {e}"))
            print(f"  [FAIL] {name}\n         {e}")
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"  [ERR ] {name}\n         {type(e).__name__}: {e}")
        else:
            passed.append(name)
            print(f"  [PASS] {name}")
    print()
    print(f"{len(passed)} passed, {len(failed)} failed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(_run_all_manual())
