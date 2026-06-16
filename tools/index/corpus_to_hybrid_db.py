"""
corpus_to_hybrid_db.py
======================

Generalized tool for turning a corpus of plain-text files into a SQLite
"hybrid FTS" database matching the schema of
``data/gamma_master_hybrid_fts_stage3.db``.

Two subcommands:

    ingest   Chunk a directory of .txt files into a new (or existing) DB.
             Populates: chunks, fulltext_fts, entities_fts.
             By default, each file's text is first run through the
             clean_chunks.py sentence-pattern filter (drops navigation
             chrome, CTAs, cookie banners, etc.) before word-window
             chunking. Pass --no-clean to skip cleaning.
             By default also runs spaCy NER and populates
             entities / chunk_entities and the entities portion of
             entities_text.
             Pass --with-topics to additionally run the BGE+LLM topic
             classifier from rag_classifier_pipeline6.py.

    enrich   Operate on an existing DB.  Re-runs NER and/or topic
             classification over chunks already present.  Useful for
             filling these in after a bare-bones ingest, or for
             regenerating them with --rebuild.

Examples
--------

    # Bare chunks + spaCy NER, with a filename->source_url CSV mapping
    python corpus_to_hybrid_db.py ingest \\
        --corpus /data/trineday_txt \\
        --db    /data/trineday.db \\
        --subset-name "Trine Day" \\
        --url-map /data/trineday_urls_map.csv

    # Same, but skip NER (chunks + FTS only)
    python corpus_to_hybrid_db.py ingest \\
        --corpus /data/some_txt \\
        --db /data/some.db \\
        --subset-name "Some Source" \\
        --no-ner

    # Full pipeline (chunks + NER + topic classification)
    python corpus_to_hybrid_db.py ingest \\
        --corpus /data/some_txt \\
        --db /data/some.db \\
        --subset-name "Some Source" \\
        --with-topics

    # Run NER + topic classification over an existing DB (no re-chunking)
    python corpus_to_hybrid_db.py enrich \\
        --db /data/some.db \\
        --entities --topics --rebuild

Schema produced
---------------

Identical to ``data/gamma_master_hybrid_fts_stage3.db``:

    chunks(lookup_id PK, chunk_id, title, subset_name, domain,
           source_url, entities_text, fulltext_text)
    entities(entity_id PK AUTOINC, canonical_name, type,
             UNIQUE(canonical_name, type))
    topics(topic_id PK AUTOINC, domain, topic_name,
           UNIQUE(domain, topic_name))
    chunk_entities(chunk_lookup_id, entity_id, PK both, FKs)
    chunk_topics(chunk_lookup_id, topic_id,   PK both, FKs)

    entities_fts  FTS5(entities_text, tokenize='unicode61')
    fulltext_fts  FTS5(title, fulltext_text, tokenize='unicode61')

    indexes:
      idx_chunk_entities_entity_id
      idx_chunk_topics_topic_id
      idx_entities_name (on canonical_name)

FTS rowids correspond 1:1 to chunks.lookup_id.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# Make sibling-module imports work whether the script is run directly
# (python tools/index/corpus_to_hybrid_db.py ...) or imported.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# clean_chunks.py lives in the sibling tools/clean/ directory (the tools/
# tree is organized by pipeline stage: ingest / clean / index / eval).
# Put that directory on the path so the delegation import below resolves.
_CLEAN_DIR = _HERE.parent / "clean"
if str(_CLEAN_DIR) not in sys.path:
    sys.path.insert(0, str(_CLEAN_DIR))

# clean_chunks.py is the canonical home of the low-value-sentence patterns
# and the clean_text() routine. We delegate to it so the two tools stay
# in lock-step — add a new pattern there and it shows up here, too.
try:
    from clean_chunks import (                            # type: ignore[import-not-found]
        clean_text as _clean_text_impl,
        PATTERNS as _CLEAN_PATTERNS,
    )
    _CLEAN_AVAILABLE = True
except ImportError as _e:
    _clean_text_impl = None                              # type: ignore[assignment]
    _CLEAN_PATTERNS = []                                  # type: ignore[assignment]
    _CLEAN_AVAILABLE = False
    _CLEAN_IMPORT_ERR = _e

# --------------------------------------------------------------------------- #
# Constants / defaults
# --------------------------------------------------------------------------- #

DEFAULT_CHUNK_WORDS = 480
DEFAULT_OVERLAP_WORDS = 80
DEFAULT_MIN_WORDS = 80
DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_BGE_MODEL = "BAAI/bge-large-en"
DEFAULT_LLM_BATCH_SIZE = 32
COMMIT_BATCH_SIZE = 500          # how many chunk rows per transaction
NER_BATCH_SIZE = 32              # spaCy .pipe() batch size

# spaCy label -> our entity "type"
LABEL_TYPE_MAP = {
    "PERSON":      "person",
    "ORG":         "organization",
    "GPE":         "location",
    "LOC":         "location",
    "WORK_OF_ART": "work",
    "EVENT":       "event",
    "DATE":        "date",
}

IGNORE_LABELS = {"CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "TIME", "MONEY"}

PERSON_WHITELIST = {"Hitler", "Nixon", "Stalin", "Lenin"}
BLACKLIST = {"Darth Vader"}

# Per-chunk caps for each entity bucket (matches rag_classifier_pipeline6.py)
ENTITY_LIMITS = {
    "person":       5,
    "organization": 4,
    "location":     3,
    "work":         None,   # no cap
    "event":        None,
    "date":         None,
}


# --------------------------------------------------------------------------- #
# Text + chunk helpers (lifted from trineday-rag.ipynb, with small tweaks)
# --------------------------------------------------------------------------- #

def read_text_file(path: Path) -> str:
    """Read a file as UTF-8 text, replacing undecodable bytes.

    Args:
        path: Path to the text file.

    Returns:
        The file's contents decoded as UTF-8, with malformed bytes
        replaced rather than raising.
    """
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def normalize_whitespace(s: str) -> str:
    """Normalize line endings and collapse excess whitespace.

    Converts CRLF/CR to LF, collapses runs of 3+ blank lines to a single
    blank line and runs of 2+ spaces/tabs to one space.

    Args:
        s: Raw text.

    Returns:
        The whitespace-normalized, stripped text.
    """
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def sha1_short(s: str, n: int = 12) -> str:
    """Return the first ``n`` hex characters of the SHA-1 of ``s``.

    Args:
        s: String to hash (encoded UTF-8, undecodable bytes ignored).
        n: Number of leading hex digits to keep.

    Returns:
        A short hex digest used as a content fingerprint.
    """
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:n]


def chunk_text_words(
    text: str,
    chunk_words: int,
    overlap_words: int,
    min_words: int,
) -> List[str]:
    """Word-window chunker (same algorithm as the notebook)."""
    words = text.split()
    if len(words) <= chunk_words:
        return [text.strip()] if len(words) >= min_words else []

    chunks: List[str] = []
    step = max(1, chunk_words - overlap_words)
    for start in range(0, len(words), step):
        end = start + chunk_words
        window = words[start:end]
        if len(window) < min_words:
            break
        chunks.append(" ".join(window).strip())
        if end >= len(words):
            break
    return chunks


def load_url_map(csv_path: Path) -> Dict[str, str]:
    """CSV with columns: filename,source_url."""
    out: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = (row.get("filename") or "").strip()
            url = (row.get("source_url") or "").strip()
            if fn:
                out[fn] = url
    return out


def list_txt_files(folder: Path, recursive: bool = False) -> List[Path]:
    """List ``.txt`` files in a folder, sorted by path.

    Args:
        folder: Directory to scan.
        recursive: If True, descend into subdirectories.

    Returns:
        Sorted list of ``.txt`` file paths.

    Raises:
        FileNotFoundError: If ``folder`` does not exist.
    """
    if not folder.exists():
        raise FileNotFoundError(folder)
    pattern = "**/*.txt" if recursive else "*.txt"
    return sorted(p for p in folder.glob(pattern) if p.is_file())


# --------------------------------------------------------------------------- #
# Low-value-sentence cleaning (delegates to clean_chunks.PATTERNS)
# --------------------------------------------------------------------------- #

@dataclass
class CleanStats:
    """Aggregated cleaning stats over a corpus pass."""
    files_examined:    int                = 0
    files_modified:    int                = 0
    sentences_removed: int                = 0
    chars_kept:        int                = 0
    chars_removed:     int                = 0
    pattern_hits:      Counter            = field(default_factory=Counter)
    examples:          Dict[str, List[str]] = field(default_factory=dict)
    examples_cap:      int                = 3

    def init_buckets(self, pattern_names: Iterable[str]) -> None:
        """Ensure an empty example bucket exists for each pattern name.

        Args:
            pattern_names: Names of cleaning patterns to register.
        """
        for name in pattern_names:
            self.examples.setdefault(name, [])

    def report(self) -> str:
        """Render a human-readable summary of the cleaning pass.

        Returns:
            A multi-line string with file/sentence/char counts, net text
            reduction percentage, and per-pattern hit counts with example
            removed sentences.
        """
        lines = []
        lines.append("Cleaning summary")
        lines.append("-" * 60)
        lines.append(f"Files examined        : {self.files_examined:,}")
        lines.append(f"Files modified        : {self.files_modified:,}")
        lines.append(f"Sentences removed     : {self.sentences_removed:,}")
        lines.append(f"Chars kept            : {self.chars_kept:,}")
        lines.append(f"Chars removed         : {self.chars_removed:,}")
        total = self.chars_kept + self.chars_removed
        if total:
            pct = 100.0 * self.chars_removed / total
            lines.append(f"Net text reduction    : {pct:.2f}%")
        lines.append("")
        lines.append("Per-pattern hits (sorted desc):")
        if not self.pattern_hits:
            lines.append("  (no patterns matched any sentence)")
        else:
            for name, n in self.pattern_hits.most_common():
                lines.append(f"  {n:8,d}  {name}")
                for ex in self.examples.get(name, []):
                    lines.append(f"            ex: {ex}")
        return "\n".join(lines)


def clean_file_text(text: str, stats: Optional[CleanStats] = None) -> str:
    """
    Apply the clean_chunks.py sentence-pattern filter to one file's text.
    Sentence boundaries are pragmatic (`.!?` + whitespace, or newline);
    sentences matching any blocklist pattern are dropped, the rest are
    re-stitched with single spaces. Stats are accumulated on `stats` if
    supplied.
    """
    if not _CLEAN_AVAILABLE:
        raise RuntimeError(
            "Cleaning requested but clean_chunks.py could not be imported: "
            f"{_CLEAN_IMPORT_ERR!r}. Either place clean_chunks.py next to "
            "this script, or pass --no-clean."
        )
    if not text:
        return text
    original_len = len(text)
    cleaned, hits, removed = _clean_text_impl(text)
    if stats is not None:
        stats.files_examined += 1
        if hits and cleaned != text:
            stats.files_modified += 1
            stats.pattern_hits.update(hits)
            stats.sentences_removed += len(removed)
            stats.chars_kept += len(cleaned)
            stats.chars_removed += (original_len - len(cleaned))
            stats.init_buckets(name for name, _ in _CLEAN_PATTERNS)
            for name, sent in removed:
                bucket = stats.examples.setdefault(name, [])
                if len(bucket) < stats.examples_cap:
                    bucket.append(sent[:200])
        else:
            stats.chars_kept += original_len
    return cleaned


# --------------------------------------------------------------------------- #
# Entity cleanup helpers (mirrors rag_classifier_pipeline6.py)
# --------------------------------------------------------------------------- #

def clean_entity(text: str) -> str:
    """Normalize an entity surface form for deduplication.

    Strips surrounding whitespace, removes a trailing possessive ``'s``,
    and collapses internal whitespace to single spaces.

    Args:
        text: Raw entity text from the NER model.

    Returns:
        The cleaned entity string.
    """
    text = text.strip()
    text = re.sub(r"[’']s$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def is_strong_person(name: str) -> bool:
    """Decide whether a PERSON entity is specific enough to keep.

    Accepts any name on ``PERSON_WHITELIST`` (notable single-token
    figures); otherwise requires at least two whitespace-separated tokens
    so bare first/last names are dropped.

    Args:
        name: Candidate person name.

    Returns:
        True if the name should be retained.
    """
    if name in PERSON_WHITELIST:
        return True
    return len(name.split()) >= 2


def canonicalize_persons(persons: Iterable[str]) -> List[str]:
    """Collapse person mentions sharing a last name to the longest form.

    Groups names by their final token and keeps the longest surface form
    per group (e.g. "Nixon" and "Richard Nixon" collapse to the latter).

    Args:
        persons: Iterable of cleaned person names.

    Returns:
        Sorted list of canonical person names, one per last name.
    """
    canonical: Dict[str, str] = {}
    for p in persons:
        parts = p.split()
        if not parts:
            continue
        last = parts[-1]
        if last not in canonical or len(p) > len(canonical[last]):
            canonical[last] = p
    return sorted(canonical.values())


def canonicalize_orgs(orgs: Iterable[str]) -> List[str]:
    """Deduplicate organization mentions case- and ``the``-insensitively.

    Strips a leading "the ", lower-cases for keying, and keeps the longest
    surface form per key.

    Args:
        orgs: Iterable of cleaned organization names.

    Returns:
        Sorted list of canonical organization names.
    """
    canonical: Dict[str, str] = {}
    for o in orgs:
        o = re.sub(r"^the\s+", "", o, flags=re.I)
        key = o.lower()
        if key not in canonical or len(o) > len(canonical[key]):
            canonical[key] = o
    return sorted(set(canonical.values()))


def filter_dates(dates: Iterable[str]) -> List[str]:
    """Keep only DATE mentions that contain a 4-digit year.

    Args:
        dates: Iterable of cleaned date strings.

    Returns:
        Sorted, deduplicated list of dates beginning with a 4-digit year.
    """
    keep = []
    for d in dates:
        if re.match(r"\b\d{4}\b", d):
            keep.append(d)
    return sorted(set(keep))


def group_entities(raw_entities: Iterable[Dict[str, str]]) -> Dict[str, List[str]]:
    """
    raw_entities: iterable of {"text": ..., "label": ...} dicts (spaCy-style).
    Returns dict keyed by our entity "type" (person/organization/location/...)
    with cleaned, deduplicated, capped lists of canonical names.
    """
    buckets: Dict[str, List[str]] = {t: [] for t in {"person", "organization", "location", "work", "event", "date"}}

    for ent in raw_entities:
        label = ent.get("label", "")
        if label in IGNORE_LABELS:
            continue
        bucket = LABEL_TYPE_MAP.get(label)
        if not bucket:
            continue
        text = clean_entity(ent.get("text", ""))
        if not text or text in BLACKLIST:
            continue
        buckets[bucket].append(text)

    persons = [p for p in buckets["person"] if is_strong_person(p)]
    persons = canonicalize_persons(persons)
    orgs = canonicalize_orgs(buckets["organization"])
    locations = sorted(set(buckets["location"]))
    works = sorted(set(buckets["work"]))
    events = sorted(set(buckets["event"]))
    dates = filter_dates(buckets["date"])

    grouped = {
        "person":       persons,
        "organization": orgs,
        "location":     locations,
        "work":         works,
        "event":        events,
        "date":         dates,
    }
    # Apply per-bucket caps
    for k, cap in ENTITY_LIMITS.items():
        if cap is not None:
            grouped[k] = grouped[k][:cap]
    # Drop empty buckets
    return {k: v for k, v in grouped.items() if v}


# --------------------------------------------------------------------------- #
# entities_text composition
# --------------------------------------------------------------------------- #

def compose_entities_text(
    grouped_entities: Dict[str, List[str]],
    topics: Sequence[str] = (),
) -> str:
    """
    Build the lowercase, space-separated 'entities_text' column used by
    entities_fts.  Order (matching the existing master DB):
        persons, organizations, locations, works, events, dates,
        then topic names (underscores -> spaces).
    """
    order = ["person", "organization", "location", "work", "event", "date"]
    parts: List[str] = []
    for k in order:
        for v in grouped_entities.get(k, []):
            parts.append(v.lower())
    for t in topics:
        parts.append(t.replace("_", " ").lower())
    return " ".join(parts).strip()


# --------------------------------------------------------------------------- #
# SQLite schema setup
# --------------------------------------------------------------------------- #

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

CREATE TABLE IF NOT EXISTS topics (
    topic_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    domain     TEXT NOT NULL,
    topic_name TEXT NOT NULL,
    UNIQUE(domain, topic_name)
);

CREATE TABLE IF NOT EXISTS chunk_entities (
    chunk_lookup_id INTEGER NOT NULL,
    entity_id       INTEGER NOT NULL,
    PRIMARY KEY (chunk_lookup_id, entity_id),
    FOREIGN KEY (chunk_lookup_id) REFERENCES chunks(lookup_id)   ON DELETE CASCADE,
    FOREIGN KEY (entity_id)       REFERENCES entities(entity_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS chunk_topics (
    chunk_lookup_id INTEGER NOT NULL,
    topic_id        INTEGER NOT NULL,
    PRIMARY KEY (chunk_lookup_id, topic_id),
    FOREIGN KEY (chunk_lookup_id) REFERENCES chunks(lookup_id) ON DELETE CASCADE,
    FOREIGN KEY (topic_id)        REFERENCES topics(topic_id)  ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_chunk_entities_entity_id
    ON chunk_entities(entity_id);
CREATE INDEX IF NOT EXISTS idx_chunk_topics_topic_id
    ON chunk_topics(topic_id);
CREATE INDEX IF NOT EXISTS idx_entities_name
    ON entities(canonical_name);

CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
    entities_text,
    tokenize='unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS fulltext_fts USING fts5(
    title,
    fulltext_text,
    tokenize='unicode61'
);
"""


def open_db(db_path: Path, *, create: bool = True) -> sqlite3.Connection:
    """Open the hybrid-FTS SQLite DB, applying pragmas and the schema.

    Enables foreign keys and WAL journaling, then runs ``SCHEMA_SQL``
    (all ``CREATE ... IF NOT EXISTS``) so an existing DB is left intact
    and a new one is fully initialized.

    Args:
        db_path: Path to the SQLite database file.
        create: If False, require the file to already exist.

    Returns:
        An open SQLite connection.

    Raises:
        FileNotFoundError: If ``create`` is False and the file is missing.
    """
    if not create and not db_path.exists():
        raise FileNotFoundError(db_path)
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    con.executescript(SCHEMA_SQL)
    return con


@contextmanager
def transaction(con: sqlite3.Connection):
    """Context manager that commits on success and rolls back on error.

    Args:
        con: The SQLite connection to commit or roll back.

    Raises:
        BaseException: Re-raises any exception after rolling back.
    """
    try:
        yield
        con.commit()
    except BaseException:
        con.rollback()
        raise


# --------------------------------------------------------------------------- #
# Caches keyed lookup tables to avoid per-row SELECTs
# --------------------------------------------------------------------------- #

class EntityCache:
    """Pre-loads (canonical_name, type) -> entity_id from disk and lazily
       inserts new entities, returning the assigned entity_id."""
    def __init__(self, con: sqlite3.Connection):
        """Pre-load the (canonical_name, type) -> entity_id map from disk.

        Args:
            con: Open SQLite connection to read existing entities from.
        """
        self.con = con
        self._cache: Dict[Tuple[str, str], int] = {}
        for eid, name, typ in con.execute(
            "SELECT entity_id, canonical_name, type FROM entities"
        ):
            self._cache[(name, typ)] = eid

    def get_or_insert(self, canonical_name: str, type_: str) -> int:
        """Return the entity_id for a (name, type), inserting if absent.

        Checks the in-memory cache first, otherwise upserts into the
        ``entities`` table and caches the assigned id.

        Args:
            canonical_name: Canonical entity name.
            type_: Entity type (person/organization/location/...).

        Returns:
            The entity_id for the row.
        """
        key = (canonical_name, type_)
        eid = self._cache.get(key)
        if eid is not None:
            return eid
        cur = self.con.execute(
            "INSERT INTO entities(canonical_name, type) VALUES(?, ?) "
            "ON CONFLICT(canonical_name, type) DO UPDATE SET canonical_name=excluded.canonical_name "
            "RETURNING entity_id",
            (canonical_name, type_),
        )
        eid = cur.fetchone()[0]
        self._cache[key] = eid
        return eid


class TopicCache:
    """Same idea for (domain, topic_name) -> topic_id."""
    def __init__(self, con: sqlite3.Connection):
        """Pre-load the (domain, topic_name) -> topic_id map from disk.

        Args:
            con: Open SQLite connection to read existing topics from.
        """
        self.con = con
        self._cache: Dict[Tuple[str, str], int] = {}
        for tid, domain, tname in con.execute(
            "SELECT topic_id, domain, topic_name FROM topics"
        ):
            self._cache[(domain, tname)] = tid

    def get_or_insert(self, domain: str, topic_name: str) -> int:
        """Return the topic_id for a (domain, name), inserting if absent.

        Checks the in-memory cache first, otherwise upserts into the
        ``topics`` table and caches the assigned id.

        Args:
            domain: Topic domain.
            topic_name: Topic name within the domain.

        Returns:
            The topic_id for the row.
        """
        key = (domain, topic_name)
        tid = self._cache.get(key)
        if tid is not None:
            return tid
        cur = self.con.execute(
            "INSERT INTO topics(domain, topic_name) VALUES(?, ?) "
            "ON CONFLICT(domain, topic_name) DO UPDATE SET domain=excluded.domain "
            "RETURNING topic_id",
            (domain, topic_name),
        )
        tid = cur.fetchone()[0]
        self._cache[key] = tid
        return tid


# --------------------------------------------------------------------------- #
# Lazy NER / topic-classifier loaders (so the tool runs without torch when
# you don't need the heavy stuff)
# --------------------------------------------------------------------------- #

def load_spacy(model_name: str):
    """Lazily import spaCy and load an NER-only pipeline.

    Disables the tagger, parser, lemmatizer, and attribute ruler so only
    the NER component runs, for speed.

    Args:
        model_name: spaCy model to load (e.g. ``en_core_web_sm``).

    Returns:
        The loaded spaCy ``Language`` object.

    Raises:
        RuntimeError: If the model is not installed, with download hint.
    """
    import spacy
    try:
        return spacy.load(model_name, disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])
    except OSError as e:
        raise RuntimeError(
            f"spaCy model {model_name!r} not installed.  Run:\n"
            f"    python -m spacy download {model_name}\n"
            f"(original error: {e})"
        )


def spacy_entities_for_text(nlp, text: str) -> List[Dict[str, str]]:
    """Run spaCy NER on a single text and return its entities.

    Args:
        nlp: A loaded spaCy ``Language`` pipeline.
        text: Text to extract entities from.

    Returns:
        List of ``{"text": ..., "label": ...}`` dicts, one per entity.
    """
    doc = nlp(text)
    return [{"text": e.text, "label": e.label_} for e in doc.ents]


def spacy_entities_batch(nlp, texts: List[str], batch_size: int = NER_BATCH_SIZE) -> List[List[Dict[str, str]]]:
    """Run spaCy NER over many texts via ``nlp.pipe`` for throughput.

    Args:
        nlp: A loaded spaCy ``Language`` pipeline.
        texts: Texts to extract entities from.
        batch_size: Batch size passed to ``nlp.pipe``.

    Returns:
        One entity list per input text (each a list of
        ``{"text": ..., "label": ...}`` dicts), in input order.
    """
    out: List[List[Dict[str, str]]] = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        out.append([{"text": e.text, "label": e.label_} for e in doc.ents])
    return out


# -- topic classifier ------------------------------------------------------- #

class TopicClassifier:
    """
    Wraps BGE-large (for domain selection) + Llama-3.1-8B-Instruct (for
    topic selection within the chosen domain), exactly as in
    rag_classifier_pipeline6.py.  Imports torch/transformers lazily so the
    rest of the tool runs on a vanilla Python install.
    """

    PROMPT_TEMPLATE = (
        "You are a classification system.\n\n"
        "Your task is to analyze the text and determine:\n\n"
        "1. The most appropriate DOMAIN.\n"
        "2. Up to 3 TOPICS that best describe the text.\n\n"
        "Rules:\n"
        "- DOMAIN must be chosen ONLY from the provided domain list.\n"
        "- TOPICS must be chosen ONLY from the provided topic list.\n"
        "- Return a maximum of 3 topics.\n"
        "- Do not invent new domains or topics.\n\n"
        "Allowed Domains:\n{domains}\n\n"
        "Allowed Topics:\n{topics}\n\n"
        "Return JSON ONLY in this format:\n\n"
        "{{\n  \"domain\": \"...\",\n  \"topics\": [\"...\", \"...\"],\n  \"confidence\": <float>\n}}\n\n"
        "Text:\n{text}\n\n"
        "You MUST return valid JSON.\n"
        "If invalid JSON is produced, the response is incorrect.\n"
        "Do not include any text before or after JSON.\n"
    )

    def __init__(
        self,
        topics_module_path: Optional[Path] = None,
        llm_model: str = DEFAULT_LLM_MODEL,
        bge_model: str = DEFAULT_BGE_MODEL,
    ):
        """Load the topics taxonomy and initialize the BGE + LLM models.

        Args:
            topics_module_path: Optional path to a ``rag_topics.py`` module
                defining ``DOMAINS`` and ``TOPICS``.
            llm_model: HF model id for the topic-selection LLM.
            bge_model: HF model id for the BGE domain-embedding model.
        """
        self.llm_model_name = llm_model
        self.bge_model_name = bge_model
        self.DOMAINS, self.TOPICS = self._load_topics_module(topics_module_path)
        self._init_models()

    def _load_topics_module(self, topics_module_path: Optional[Path]):
        """Load the ``DOMAINS`` and ``TOPICS`` taxonomy definitions.

        Resolution order: the given path, then the default
        ``rag_topics.py`` under peers_dev/preprocessing, then an importable
        ``rag_topics`` on ``PYTHONPATH``.

        Args:
            topics_module_path: Optional explicit path to the module.

        Returns:
            A ``(DOMAINS, TOPICS)`` tuple from the resolved module.
        """
        # Try given path, then peers_dev/preprocessing/rag_topics.py, then
        # importable rag_topics on PYTHONPATH.
        candidates = []
        if topics_module_path is not None:
            candidates.append(Path(topics_module_path))
        candidates.append(Path("/Volumes/SSK/peers_dev/preprocessing/rag_topics.py"))
        for c in candidates:
            if c and c.exists():
                ns: Dict[str, object] = {}
                exec(compile(c.read_text(encoding="utf-8"), str(c), "exec"), ns)
                return ns["DOMAINS"], ns["TOPICS"]
        # Fallback: try import
        import rag_topics  # type: ignore
        return rag_topics.DOMAINS, rag_topics.TOPICS

    def _init_models(self):
        """Lazily load the BGE embedder and the 4-bit quantized LLM.

        Imports torch/transformers/sentence-transformers, encodes each
        domain description into a normalized embedding matrix, and loads
        the causal-LM tokenizer and 4-bit (nf4) quantized chat model.
        """
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.np = np

        self.bge = SentenceTransformer(self.bge_model_name)
        self.domain_vectors = {
            d: self.bge.encode(desc, normalize_embeddings=True)
            for d, desc in self.DOMAINS.items()
        }
        self.domain_keys = list(self.domain_vectors.keys())
        self.domain_matrix = np.array([self.domain_vectors[d] for d in self.domain_keys])

        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self.chat = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            torch_dtype="auto",
            quantization_config=bnb,
            device_map="auto",
        )

    # ----- domain via BGE ----------------------------------------------------
    def classify_domain_batch(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        """Pick the best-matching domain for each text via BGE cosine sim.

        Encodes texts with BGE and takes the argmax cosine similarity
        against the precomputed domain-description embeddings.

        Args:
            texts: Texts to classify.

        Returns:
            A ``(domains, scores)`` tuple of parallel lists: the chosen
            domain key and its similarity score for each input.
        """
        vecs = self.bge.encode(texts, normalize_embeddings=True)
        sims = vecs @ self.domain_matrix.T
        domains, scores = [], []
        for row in sims:
            idx = int(self.np.argmax(row))
            domains.append(self.domain_keys[idx])
            scores.append(float(row[idx]))
        return domains, scores

    # ----- LLM batch ---------------------------------------------------------
    def _generate_batch(self, prompts: List[str], max_new_tokens: int = 120) -> List[str]:
        """Run greedy LLM generation over a batch of prompts.

        Wraps each prompt in a system/user chat template, left-pads and
        truncates the batch, generates deterministically (no sampling),
        and decodes only the newly generated tokens.

        Args:
            prompts: User prompts to classify.
            max_new_tokens: Generation cap per prompt.

        Returns:
            One decoded completion string per prompt.
        """
        messages = [
            [
                {"role": "system", "content": "You are a strict JSON classification system."},
                {"role": "user", "content": p},
            ]
            for p in prompts
        ]
        rendered = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.chat.device)
        outputs = self.chat.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        results: List[str] = []
        prompt_len = inputs["input_ids"].shape[1]
        for i in range(len(prompts)):
            gen = outputs[i][prompt_len:]
            results.append(self.tokenizer.decode(gen, skip_special_tokens=True).strip())
        return results

    # ----- classify a batch of chunk texts -----------------------------------
    def classify_batch(self, texts: List[str]) -> List[Dict[str, object]]:
        """Classify chunk texts into a domain and up to three topics.

        Truncates long texts (keeping head and tail), selects a domain via
        BGE, prompts the LLM for topics within that domain, parses the JSON
        (falling back to the first few allowed topics on failure), and
        validates the returned topics against the domain's allowed list.

        Args:
            texts: Chunk texts to classify.

        Returns:
            One dict per text with keys ``domain`` (str), ``topics``
            (list of <=3 allowed topic names), and ``confidence`` (float).
        """
        truncated = [
            (t[:700] + "\n...\n" + t[-700:]) if len(t) > 1400 else t
            for t in texts
        ]
        domains, scores = self.classify_domain_batch(truncated)

        prompts: List[str] = []
        for text, domain in zip(truncated, domains):
            topic_list = self.TOPICS.get(domain, {})
            prompts.append(
                self.PROMPT_TEMPLATE.format(
                    domains=domain,
                    topics=", ".join(topic_list.keys()),
                    text=text,
                )
            )
        raws = self._generate_batch(prompts)

        results: List[Dict[str, object]] = []
        for domain, score, raw in zip(domains, scores, raws):
            try:
                data = _safe_parse_json(raw)
                topics = _normalize_topics(data.get("topics", []))
                domain_out = data.get("domain", domain) or domain
                conf = float(data.get("confidence", score))
            except Exception:
                topics = list(self.TOPICS.get(domain, {}).keys())[:3]
                domain_out = domain
                conf = 0.5
            # Validate topics against allowed list for the chosen domain
            allowed = set(self.TOPICS.get(domain_out, {}).keys())
            topics = [t for t in topics if t in allowed][:3]
            results.append({"domain": domain_out, "topics": topics, "confidence": conf})
        return results


def _safe_parse_json(text: str):
    """Parse JSON, retrying once with a repaired string on failure.

    Args:
        text: Raw model output expected to contain JSON.

    Returns:
        The parsed JSON object.

    Raises:
        json.JSONDecodeError: If parsing fails even after repair.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_repair_json(text))


def _repair_json(text: str) -> str:
    """Best-effort repair of truncated/malformed LLM JSON output.

    Trims any trailing text after the last ``}`` and closes an unbalanced
    quote, bracket, or brace so the result is more likely to parse.

    Args:
        text: Raw model output.

    Returns:
        The repaired string (not guaranteed to be valid JSON).
    """
    last_brace = text.rfind("}")
    if last_brace != -1:
        text = text[:last_brace + 1]
    if text.count('"') % 2 == 1:
        text += '"'
    if text.count("[") > text.count("]"):
        text += "]"
    if text.count("{") > text.count("}"):
        text += "}"
    return text


def _normalize_topics(topics) -> List[str]:
    """Deduplicate topic strings, preserving order and capping at 10.

    Drops non-string and duplicate entries.

    Args:
        topics: Raw topics value from parsed model output (may be None).

    Returns:
        Up to 10 unique topic strings in first-seen order.
    """
    seen, out = set(), []
    for t in topics or []:
        if isinstance(t, str) and t not in seen:
            seen.add(t)
            out.append(t)
    return out[:10]


# --------------------------------------------------------------------------- #
# Ingest pipeline
# --------------------------------------------------------------------------- #

def iter_corpus_chunks(
    files: List[Path],
    subset_name: str,
    *,
    url_map: Optional[Dict[str, str]],
    chunk_words: int,
    overlap_words: int,
    min_words: int,
    dedupe: bool,
    default_domain: str = "",
    clean: bool = True,
    clean_stats: Optional[CleanStats] = None,
) -> Iterator[Dict[str, object]]:
    """
    Stream chunk-dicts ready for DB insertion.

    Per-file order of operations:
        read_text_file -> normalize_whitespace -> [clean_file_text] ->
        chunk_text_words -> [sha1 dedupe] -> yield

    Cleaning happens BEFORE chunking so navigation chrome / CTAs /
    cookie banners don't end up consuming chunk budget. With clean=True
    and no clean_stats provided, cleaning runs without accumulating
    reportable stats.
    """
    seen_text_hashes = set() if dedupe else None
    for path in files:
        raw = read_text_file(path)
        text = normalize_whitespace(raw)
        if clean and text:
            text = clean_file_text(text, stats=clean_stats)
        if not text:
            continue
        chunks = chunk_text_words(text, chunk_words, overlap_words, min_words)
        if not chunks:
            continue
        title = path.stem
        source_url = (url_map or {}).get(path.name, "")
        for idx, ch in enumerate(chunks):
            if dedupe:
                h = sha1_short(ch, n=20)
                if h in seen_text_hashes:  # type: ignore[union-attr]
                    continue
                seen_text_hashes.add(h)    # type: ignore[union-attr]
            chunk_id = f"{subset_name}:{path.name}:{idx}:{sha1_short(ch)}"
            yield {
                "chunk_id":      chunk_id,
                "title":         title,
                "subset_name":   subset_name,
                "domain":        default_domain,
                "source_url":    source_url,
                "filename":      path.name,
                "text":          ch,
            }


def insert_chunk_row(
    con: sqlite3.Connection,
    row: Dict[str, object],
    *,
    grouped_entities: Optional[Dict[str, List[str]]] = None,
    topics: Sequence[Tuple[str, str]] = (),   # iterable of (domain, topic_name)
    entity_cache: Optional[EntityCache] = None,
    topic_cache: Optional[TopicCache] = None,
) -> int:
    """Insert a single chunk plus its FTS / join rows.  Returns lookup_id."""
    grouped = grouped_entities or {}
    topic_names = [tn for _, tn in topics]
    entities_text = compose_entities_text(grouped, topic_names)
    fulltext_text = (str(row["title"]) + " " + str(row["text"])).strip()
    domain = str(topics[0][0]) if topics else str(row.get("domain", "") or "")

    cur = con.execute(
        "INSERT INTO chunks(chunk_id, title, subset_name, domain, source_url, "
        "                   entities_text, fulltext_text) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            row["chunk_id"], row["title"], row["subset_name"], domain,
            row["source_url"], entities_text, fulltext_text,
        ),
    )
    lookup_id = cur.lastrowid

    # FTS rows must use the same rowid as chunks.lookup_id
    con.execute(
        "INSERT INTO entities_fts(rowid, entities_text) VALUES(?, ?)",
        (lookup_id, entities_text),
    )
    con.execute(
        "INSERT INTO fulltext_fts(rowid, title, fulltext_text) VALUES(?, ?, ?)",
        (lookup_id, row["title"], fulltext_text),
    )

    # chunk_entities
    if entity_cache is not None and grouped:
        for typ, names in grouped.items():
            for name in names:
                eid = entity_cache.get_or_insert(name, typ)
                con.execute(
                    "INSERT OR IGNORE INTO chunk_entities(chunk_lookup_id, entity_id) VALUES(?, ?)",
                    (lookup_id, eid),
                )

    # chunk_topics
    if topic_cache is not None and topics:
        for domain_, topic_name in topics:
            tid = topic_cache.get_or_insert(domain_, topic_name)
            con.execute(
                "INSERT OR IGNORE INTO chunk_topics(chunk_lookup_id, topic_id) VALUES(?, ?)",
                (lookup_id, tid),
            )

    return lookup_id


def cmd_ingest(args: argparse.Namespace) -> int:
    """Run the ``ingest`` subcommand: corpus of .txt files -> hybrid DB.

    Scans the corpus directory, optionally loads a filename->URL map,
    opens/creates the DB, optionally sets up cleaning, spaCy NER, and the
    topic classifier, then streams chunks in batches into the chunks /
    FTS / entity / topic tables. Prints progress and an optional cleaning
    report.

    Args:
        args: Parsed CLI arguments for the ingest subcommand.

    Returns:
        Process exit code (0 on success, 2 if no .txt files were found).
    """
    corpus_dir = Path(args.corpus)
    db_path = Path(args.db)
    files = list_txt_files(corpus_dir, recursive=args.recursive)
    if not files:
        print(f"No .txt files found under {corpus_dir}", file=sys.stderr)
        return 2

    url_map: Optional[Dict[str, str]] = None
    if args.url_map:
        url_map = load_url_map(Path(args.url_map))
        print(f"Loaded {len(url_map)} url mappings from {args.url_map}")

    if db_path.exists() and args.clean_existing:
        db_path.unlink()
        print(f"Removed existing DB at {db_path}")

    con = open_db(db_path, create=True)
    entity_cache = EntityCache(con)
    topic_cache = TopicCache(con)

    # NER setup
    nlp = None
    if args.ner:
        print(f"Loading spaCy model: {args.spacy_model}")
        nlp = load_spacy(args.spacy_model)

    # Topic classifier setup
    classifier: Optional[TopicClassifier] = None
    if args.with_topics:
        print("Loading topic classifier (BGE + LLM)...")
        classifier = TopicClassifier(
            topics_module_path=(Path(args.topics_module) if args.topics_module else None),
            llm_model=args.llm_model,
        )

    # Cleaning setup
    clean_stats: Optional[CleanStats] = None
    if args.clean:
        if not _CLEAN_AVAILABLE:
            print(
                f"WARNING: --clean requested but clean_chunks.py is not importable "
                f"({_CLEAN_IMPORT_ERR!r}). Continuing without cleaning.",
                file=sys.stderr,
            )
        else:
            clean_stats = CleanStats(examples_cap=args.clean_examples)
            print(f"Cleaning enabled ({len(_CLEAN_PATTERNS)} patterns from clean_chunks.py)")

    # Stream chunks; gather small batches for NER/topic passes
    chunk_iter = iter_corpus_chunks(
        files,
        subset_name=args.subset_name,
        url_map=url_map,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
        min_words=args.min_words,
        dedupe=args.dedupe,
        default_domain=args.domain or "",
        clean=args.clean and _CLEAN_AVAILABLE,
        clean_stats=clean_stats,
    )

    batch_size = max(args.batch_size, 1)
    inserted = 0
    t0 = time.perf_counter()
    buffer: List[Dict[str, object]] = []

    def _flush(buf: List[Dict[str, object]]):
        """Run NER/topics over a buffer of chunk-dicts and insert them.

        Runs spaCy NER and the topic classifier (when enabled) over the
        batch, then inserts each chunk plus its FTS and join rows inside a
        single transaction. Updates the ``inserted`` counter and clears
        the buffer.

        Args:
            buf: Chunk-dicts (as yielded by ``iter_corpus_chunks``).
        """
        nonlocal inserted
        if not buf:
            return
        texts = [str(r["text"]) for r in buf]

        ents_per_chunk: List[Dict[str, List[str]]] = [{} for _ in buf]
        if nlp is not None:
            raw_ents = spacy_entities_batch(nlp, texts)
            ents_per_chunk = [group_entities(re_) for re_ in raw_ents]

        topics_per_chunk: List[List[Tuple[str, str]]] = [[] for _ in buf]
        if classifier is not None:
            cls = classifier.classify_batch(texts)
            for i, c in enumerate(cls):
                d = str(c.get("domain", "")) or ""
                tlist = c.get("topics") or []
                topics_per_chunk[i] = [(d, t) for t in tlist if t]

        with transaction(con):
            for row, ents, tps in zip(buf, ents_per_chunk, topics_per_chunk):
                insert_chunk_row(
                    con, row,
                    grouped_entities=ents,
                    topics=tps,
                    entity_cache=entity_cache,
                    topic_cache=topic_cache,
                )
                inserted += 1
        buf.clear()
        elapsed = time.perf_counter() - t0
        rate = inserted / elapsed if elapsed > 0 else 0.0
        print(f"  ...inserted {inserted:,} chunks ({rate:.1f}/s)")

    for row in chunk_iter:
        buffer.append(row)
        if len(buffer) >= batch_size:
            _flush(buffer)
    _flush(buffer)

    con.execute("PRAGMA optimize")
    con.close()
    print(f"Done. Wrote {inserted:,} chunks to {db_path} in {(time.perf_counter()-t0):.1f}s.")
    if clean_stats is not None:
        print()
        print(clean_stats.report())
    return 0


# --------------------------------------------------------------------------- #
# Enrich pipeline
# --------------------------------------------------------------------------- #

def _rebuild_entities_text_for_chunk(
    con: sqlite3.Connection,
    lookup_id: int,
) -> None:
    """Recompute chunks.entities_text and the matching entities_fts row
       from the current chunk_entities / chunk_topics tables."""
    grouped: Dict[str, List[str]] = {}
    for name, typ in con.execute(
        "SELECT e.canonical_name, e.type FROM chunk_entities ce "
        "JOIN entities e ON ce.entity_id = e.entity_id "
        "WHERE ce.chunk_lookup_id = ? ORDER BY e.type, e.canonical_name",
        (lookup_id,),
    ):
        grouped.setdefault(typ, []).append(name)

    topic_names: List[str] = []
    primary_domain: Optional[str] = None
    for tname, dom in con.execute(
        "SELECT t.topic_name, t.domain FROM chunk_topics ct "
        "JOIN topics t ON ct.topic_id = t.topic_id "
        "WHERE ct.chunk_lookup_id = ? ORDER BY t.topic_name",
        (lookup_id,),
    ):
        topic_names.append(tname)
        if primary_domain is None:
            primary_domain = dom

    entities_text = compose_entities_text(grouped, topic_names)
    if primary_domain:
        con.execute(
            "UPDATE chunks SET entities_text = ?, domain = ? WHERE lookup_id = ?",
            (entities_text, primary_domain, lookup_id),
        )
    else:
        con.execute(
            "UPDATE chunks SET entities_text = ? WHERE lookup_id = ?",
            (entities_text, lookup_id),
        )
    con.execute("DELETE FROM entities_fts WHERE rowid = ?", (lookup_id,))
    con.execute(
        "INSERT INTO entities_fts(rowid, entities_text) VALUES(?, ?)",
        (lookup_id, entities_text),
    )


def cmd_enrich(args: argparse.Namespace) -> int:
    """Run the ``enrich`` subcommand: re-run NER/topics on existing chunks.

    Opens an existing DB, optionally wipes the targeted entity/topic
    tables (``--rebuild``), then re-runs spaCy NER and/or the topic
    classifier over every chunk in batches, replacing each chunk's
    chunk_entities / chunk_topics rows and recomputing its entities_text.

    Args:
        args: Parsed CLI arguments for the enrich subcommand.

    Returns:
        Process exit code (0 on success, 2 if neither --entities nor
        --topics was requested).
    """
    db_path = Path(args.db)
    con = open_db(db_path, create=False)

    do_entities = bool(args.entities)
    do_topics = bool(args.topics)
    if not (do_entities or do_topics):
        print("Nothing to do (pass --entities and/or --topics).", file=sys.stderr)
        return 2

    if args.rebuild:
        with transaction(con):
            if do_entities:
                con.execute("DELETE FROM chunk_entities")
                con.execute("DELETE FROM entities")
                con.execute("DELETE FROM sqlite_sequence WHERE name='entities'")
            if do_topics:
                con.execute("DELETE FROM chunk_topics")
                con.execute("DELETE FROM topics")
                con.execute("DELETE FROM sqlite_sequence WHERE name='topics'")
        print("Rebuild mode: cleared targeted tables.")

    entity_cache = EntityCache(con)
    topic_cache = TopicCache(con)

    nlp = None
    if do_entities:
        print(f"Loading spaCy model: {args.spacy_model}")
        nlp = load_spacy(args.spacy_model)

    classifier: Optional[TopicClassifier] = None
    if do_topics:
        print("Loading topic classifier (BGE + LLM)...")
        classifier = TopicClassifier(
            topics_module_path=(Path(args.topics_module) if args.topics_module else None),
            llm_model=args.llm_model,
        )

    total = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"Enriching {total:,} chunks.")

    batch_size = max(args.batch_size, 1)
    processed = 0
    t0 = time.perf_counter()

    cur = con.execute(
        "SELECT lookup_id, title, fulltext_text FROM chunks ORDER BY lookup_id"
    )
    buffer: List[Tuple[int, str, str]] = []

    def _flush(buf: List[Tuple[int, str, str]]):
        """Re-run NER/topics over a buffer of chunk rows and update the DB.

        Runs spaCy NER and the topic classifier (when enabled) over the
        batch's fulltext, then, inside one transaction, replaces each
        chunk's chunk_entities / chunk_topics rows and recomputes its
        entities_text. Updates the ``processed`` counter and clears the
        buffer.

        Args:
            buf: Tuples of ``(lookup_id, title, fulltext_text)``.
        """
        nonlocal processed
        if not buf:
            return
        lookup_ids = [r[0] for r in buf]
        # use fulltext_text (title prefix already included) as the input
        texts = [r[2] for r in buf]

        ents_per_chunk: List[Dict[str, List[str]]] = [{} for _ in buf]
        if nlp is not None:
            raw_ents = spacy_entities_batch(nlp, texts)
            ents_per_chunk = [group_entities(re_) for re_ in raw_ents]

        topics_per_chunk: List[List[Tuple[str, str]]] = [[] for _ in buf]
        if classifier is not None:
            cls = classifier.classify_batch(texts)
            for i, c in enumerate(cls):
                d = str(c.get("domain", "")) or ""
                tlist = c.get("topics") or []
                topics_per_chunk[i] = [(d, t) for t in tlist if t]

        with transaction(con):
            for lid, ents, tps in zip(lookup_ids, ents_per_chunk, topics_per_chunk):
                if do_entities:
                    # wipe existing chunk_entities for this chunk
                    con.execute(
                        "DELETE FROM chunk_entities WHERE chunk_lookup_id = ?",
                        (lid,),
                    )
                    for typ, names in ents.items():
                        for name in names:
                            eid = entity_cache.get_or_insert(name, typ)
                            con.execute(
                                "INSERT OR IGNORE INTO chunk_entities"
                                "(chunk_lookup_id, entity_id) VALUES(?, ?)",
                                (lid, eid),
                            )

                if do_topics:
                    con.execute(
                        "DELETE FROM chunk_topics WHERE chunk_lookup_id = ?",
                        (lid,),
                    )
                    for dom, tn in tps:
                        tid = topic_cache.get_or_insert(dom, tn)
                        con.execute(
                            "INSERT OR IGNORE INTO chunk_topics"
                            "(chunk_lookup_id, topic_id) VALUES(?, ?)",
                            (lid, tid),
                        )

                _rebuild_entities_text_for_chunk(con, lid)
                processed += 1
        buf.clear()
        elapsed = time.perf_counter() - t0
        rate = processed / elapsed if elapsed > 0 else 0.0
        print(f"  ...processed {processed:,}/{total:,} ({rate:.1f}/s)")

    for row in cur:
        buffer.append(row)
        if len(buffer) >= batch_size:
            _flush(buffer)
    _flush(buffer)

    con.execute("PRAGMA optimize")
    con.close()
    print(f"Done. Enriched {processed:,} chunks in {(time.perf_counter()-t0):.1f}s.")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with the ``ingest`` and ``enrich`` subcommands.

    Returns:
        A configured ``ArgumentParser`` whose subparsers set ``func`` to
        ``cmd_ingest`` / ``cmd_enrich``.
    """
    p = argparse.ArgumentParser(
        prog="corpus_to_hybrid_db",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # -------- ingest --------
    ing = sub.add_parser("ingest", help="Build a new hybrid-FTS DB from a corpus of .txt files.")
    ing.add_argument("--corpus", required=True, help="Directory of .txt files.")
    ing.add_argument("--db", required=True, help="Output SQLite DB path.")
    ing.add_argument("--subset-name", required=True,
                     help="Name to record on every chunk (e.g. 'Trine Day').")
    ing.add_argument("--domain", default="",
                     help="Default domain to record on chunks (overridden by topic classifier when --with-topics is set).")
    ing.add_argument("--url-map", default=None,
                     help="Optional CSV with columns filename,source_url.")
    ing.add_argument("--recursive", action="store_true",
                     help="Recurse into subdirectories when scanning --corpus.")
    ing.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    ing.add_argument("--overlap-words", type=int, default=DEFAULT_OVERLAP_WORDS)
    ing.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)
    ing.add_argument("--no-dedupe", dest="dedupe", action="store_false", default=True,
                     help="Disable sha1-of-text dedupe (kept on by default).")
    ing.add_argument("--no-clean", dest="clean", action="store_false", default=True,
                     help="Skip the clean_chunks.py sentence-pattern filter "
                          "(applied to each file's text before chunking).")
    ing.add_argument("--clean-examples", type=int, default=3,
                     help="How many sentence examples to keep per pattern in "
                          "the cleaning report. Default: 3.")
    ing.add_argument("--no-ner", dest="ner", action="store_false", default=True,
                     help="Skip spaCy NER (entities tables stay empty).")
    ing.add_argument("--with-topics", action="store_true",
                     help="Also run BGE + LLM topic classifier (slow; requires GPU).")
    ing.add_argument("--spacy-model", default=DEFAULT_SPACY_MODEL)
    ing.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    ing.add_argument("--topics-module", default=None,
                     help="Path to rag_topics.py defining DOMAINS/TOPICS. "
                          "Defaults to /Volumes/SSK/peers_dev/preprocessing/rag_topics.py if present.")
    ing.add_argument("--batch-size", type=int, default=DEFAULT_LLM_BATCH_SIZE,
                     help="How many chunks to process per NER/LLM batch (also commit size).")
    ing.add_argument("--clean-existing", action="store_true",
                     help="If --db exists, delete it first.")
    ing.set_defaults(func=cmd_ingest)

    # -------- enrich --------
    enr = sub.add_parser("enrich", help="Run NER and/or topic classification over an existing DB.")
    enr.add_argument("--db", required=True, help="Existing SQLite DB path.")
    enr.add_argument("--entities", action="store_true", help="Run spaCy NER over all chunks.")
    enr.add_argument("--topics", action="store_true", help="Run topic classifier over all chunks.")
    enr.add_argument("--rebuild", action="store_true",
                     help="Wipe the relevant entity/topic tables before processing.")
    enr.add_argument("--spacy-model", default=DEFAULT_SPACY_MODEL)
    enr.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    enr.add_argument("--topics-module", default=None)
    enr.add_argument("--batch-size", type=int, default=DEFAULT_LLM_BATCH_SIZE)
    enr.set_defaults(func=cmd_enrich)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: parse arguments and dispatch to the subcommand.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).

    Returns:
        The exit code returned by the dispatched subcommand handler.
    """
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
