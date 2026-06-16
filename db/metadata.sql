CREATE TABLE IF NOT EXISTS entities (
    entity_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name  TEXT NOT NULL,
    type                       TEXT NOT NULL,

    UNIQUE(canonical_name, type)
);

CREATE INDEX idx_entities_name ON entities(canonical_name);

CREATE TABLE IF NOT EXISTS chunk_entities (
    chunk_lookup_id     INTEGER NOT NULL,
    entity_id                    INTEGER NOT NULL,

    PRIMARY KEY (chunk_lookup_id, entity_id),
    FOREIGN KEY (chunk_lookup_id)  REFERENCES chunks(lookup_id) ON DELETE CASCADE,
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE RESTRICT
);

CREATE INDEX idx_chunk_entities_entity_id ON chunk_entities(entity_id);


CREATE TABLE IF NOT EXISTS topics (
    topic_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    domain               TEXT NOT NULL,
    topic_name        TEXT NOT NULL,
    UNIQUE(domain, topic_name)
);

CREATE TABLE IF NOT EXISTS chunk_topics (
    chunk_lookup_id   INTEGER NOT NULL,
    topic_id    INTEGER NOT NULL,

    PRIMARY KEY (chunk_lookup_id, topic_id),
    FOREIGN KEY (chunk_lookup_id) REFERENCES chunks(lookup_id) ON DELETE CASCADE,
    FOREIGN KEY (topic_id) REFERENCES topics(topic_id) ON DELETE RESTRICT
);

CREATE INDEX idx_chunk_topics_topic_id ON chunk_topics(topic_id);


