SELECT
    e.canonical_name,
    e.type
FROM chunks c
JOIN chunk_entities ce
  ON c.lookup_id = ce.chunk_lookup_id
JOIN entities e
  ON ce.entity_id = e.entity_id
WHERE c.chunk_id = ?;


SELECT
    t.domain,
    t.topic_name
FROM chunks c
JOIN chunk_topics ct
  ON c.lookup_id = ct.chunk_lookup_id
JOIN topics t
  ON ct.topic_id = t.topic_id
WHERE c.chunk_id = ?;
