#!/usr/bin/env python3

"""
This script updates the chunks gamma database with metadata tables for chunks and topics.
"""
import sqlite3
import json
import sys
from pathlib import Path

DB_PATH = "./data/gamma_master_hybrid_fts_stage2.db"
JSONL_PATH = "./data/gamma_complete_dataset_stage2.jsonl"

BATCH_SIZE = 1000


# -----------------------------
# Utilities
# -----------------------------

ENTITY_GROUP_MAP = {
    "persons": "person",
    "organizations": "organization",
    "locations": "location",
    "events": "event",
    "works": "work",
    "dates": "date"
}


def normalize_entity(name: str) -> str:
    """Normalize an entity name by stripping and collapsing internal whitespace."""
    return " ".join(name.strip().split())


def normalize_topic(topic: str) -> str:
    """Normalize a topic by stripping whitespace and lowercasing."""
    return topic.strip().lower()


def normalize_domain(domain: str) -> str:
    """Normalize a domain by stripping whitespace and lowercasing."""
    return domain.strip().lower()


# -----------------------------
# DB helpers
# -----------------------------

def get_or_create_entity(cur, cache, name, etype):
    """Return the entity_id for (name, etype), inserting the row if needed.

    Args:
        cur: An open SQLite cursor.
        cache: In-memory dict mapping (name, etype) -> entity_id, updated in place.
        name: The canonical entity name.
        etype: The entity type (e.g. "person", "organization").

    Returns:
        The integer entity_id, served from the cache when available and
        otherwise fetched after an INSERT OR IGNORE into the entities table.
    """
    key = (name, etype)

    if key in cache:
        return cache[key]

    cur.execute("""
        INSERT OR IGNORE INTO entities (canonical_name, type)
        VALUES (?, ?)
    """, (name, etype))

    cur.execute("""
        SELECT entity_id
        FROM entities
        WHERE canonical_name = ?
          AND type = ?
    """, (name, etype))

    entity_id = cur.fetchone()[0]
    cache[key] = entity_id
    return entity_id


def get_or_create_topic(cur, cache, domain, topic):
    """Return the topic_id for (domain, topic), inserting the row if needed.

    Args:
        cur: An open SQLite cursor.
        cache: In-memory dict mapping (domain, topic) -> topic_id, updated in place.
        domain: The normalized domain string.
        topic: The normalized topic name.

    Returns:
        The integer topic_id, served from the cache when available and
        otherwise fetched after an INSERT OR IGNORE into the topics table.
    """
    key = (domain, topic)

    if key in cache:
        return cache[key]

    cur.execute("""
        INSERT OR IGNORE INTO topics (domain, topic_name)
        VALUES (?, ?)
    """, (domain, topic))

    cur.execute("""
        SELECT topic_id
        FROM topics
        WHERE domain = ?
          AND topic_name = ?
    """, (domain, topic))

    topic_id = cur.fetchone()[0]
    cache[key] = topic_id
    return topic_id


def build_chunk_lookup_map(cur):
    """
    chunk_id (text) -> lookup_id (int)
    Pull once into memory.
    """
    mapping = {}
    for row in cur.execute("""
        SELECT chunk_id, lookup_id
        FROM chunks
    """):
        mapping[row[0]] = row[1]
    return mapping


# -----------------------------
# Main loader
# -----------------------------

def load_metadata(jsonl_path, db_path):
    """Populate entity and topic metadata tables from a chunk dataset JSONL.

    Streams the JSONL file line by line. For each document whose chunk_id maps
    to a known chunk, it normalizes and upserts the grouped entities and topics,
    then links them to the chunk via the chunk_entities and chunk_topics join
    tables. Commits in batches of BATCH_SIZE and prints progress and summary
    counts (processed, missing, cached entities/topics).

    Args:
        jsonl_path: Path to the chunk dataset JSONL file.
        db_path: Path to the SQLite database to update.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    print("Loading chunk lookup map...")
    chunk_lookup = build_chunk_lookup_map(cur)
    print(f"Loaded {len(chunk_lookup):,} chunk mappings")

    entity_cache = {}
    topic_cache = {}

    processed = 0
    missing_chunks = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:

        for line_num, line in enumerate(f, 1):

            line = line.strip()
            if not line:
                continue

            try:
                doc = json.loads(line)
            except Exception as e:
                print(f"Bad JSON line {line_num}: {e}")
                continue

            chunk_id = doc.get("chunk_id")

            if chunk_id not in chunk_lookup:
                missing_chunks += 1
                continue

            chunk_lookup_id = chunk_lookup[chunk_id]

            #
            # --------------------
            # Entities
            # --------------------
            #
            grouped = doc.get("entities_grouped", {})

            for group_name, values in grouped.items():

                if not values:
                    continue

                etype = ENTITY_GROUP_MAP.get(
                    group_name,
                    group_name.rstrip("s")
                )

                for raw_name in values:
                    name = normalize_entity(raw_name)
                    if not name:
                        continue

                    entity_id = get_or_create_entity(
                        cur,
                        entity_cache,
                        name,
                        etype
                    )

                    cur.execute("""
                        INSERT OR IGNORE INTO chunk_entities
                        (chunk_lookup_id, entity_id)
                        VALUES (?, ?)
                    """, (
                        chunk_lookup_id,
                        entity_id
                    ))

            #
            # --------------------
            # Topics
            # --------------------
            #

            domain = normalize_domain(
                doc.get("domain", "unknown")
            )

            for raw_topic in doc.get("topics", []):
                topic_name = normalize_topic(raw_topic)

                if not topic_name:
                    continue

                topic_id = get_or_create_topic(
                    cur,
                    topic_cache,
                    domain,
                    topic_name
                )

                cur.execute("""
                    INSERT OR IGNORE INTO chunk_topics
                    (chunk_lookup_id, topic_id)
                    VALUES (?, ?)
                """, (
                    chunk_lookup_id,
                    topic_id
                ))

            processed += 1

            if processed % BATCH_SIZE == 0:
                conn.commit()
                print(
                    f"Processed {processed:,} chunks "
                    f"(missing {missing_chunks})"
                )

    conn.commit()

    print("\nDone.")
    print(f"Chunks processed: {processed:,}")
    print(f"Missing chunk_ids: {missing_chunks:,}")
    print(f"Entities cached: {len(entity_cache):,}")
    print(f"Topics cached: {len(topic_cache):,}")

    conn.close()


if __name__ == "__main__":

    json_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else JSONL_PATH
    )

    if not Path(json_path).exists():
        print(f"Missing file: {json_path}")
        sys.exit(1)

    load_metadata(
        json_path,
        DB_PATH
    )