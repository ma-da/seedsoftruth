"""Text cleaning and entity tooling for the RAG ingestion pipeline.

This module provides three related capabilities:

  * ``clean_text_for_rag`` and its helpers — strip navigation, promotional
    boilerplate, table-of-contents blocks, markdown markup, and URLs from
    scraped source text while preserving paragraph structure, optionally
    separating a trailing references section.
  * spaCy-backed named-entity extraction — ``extract_entities`` and the
    ``group_entities`` / canonicalization helpers turn raw text into a
    bucketed, deduplicated, capped entity dictionary.
  * an entity-overlap reranker — ``rerank_chunks`` and its scoring helpers
    reorder retrieved chunks by how well their entities match the query's.
"""

import math
import re
from typing import Dict, List, Optional, Tuple

import spacy

# ----------------------------
# Regex patterns
# ----------------------------

URL_RE = re.compile(r"https?://\S+")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
BOLD_RE = re.compile(r"\*\*(.*?)\*\*", flags=re.DOTALL)

TOC_LINE_RE = re.compile(r"^\s*\d+(\.\d+)*\s+.*\.{3,}\s*\d+\s*$")

REFERENCE_HEADERS = {
    "references",
    "bibliography",
    "works cited",
    "literature cited",
}

META_PATTERNS = [
    re.compile(r"WantToKnow\.info", re.IGNORECASE),
    re.compile(r"\bPEERS\b"),
    re.compile(r"click here", re.IGNORECASE),
    re.compile(r"subscribe", re.IGNORECASE),
    re.compile(r"all rights reserved", re.IGNORECASE),
]

NAV_LINE_RE = re.compile(
    r"^\s*(home|about|contact|privacy|terms)\b", re.IGNORECASE
)


# ----------------------------
# Core utilities
# ----------------------------


def normalize_newlines(text: str) -> str:
    """Normalizes line endings and collapses excessive blank runs.

    Converts Windows/Mac line endings to ``\\n`` and reduces any run of
    four or more consecutive newlines to three.

    Args:
        text: The text to normalize.

    Returns:
        The text with normalized newlines.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text


def replace_markdown_links(text: str) -> str:
    """Replaces markdown links with their visible link text.

    Args:
        text: The text to process.

    Returns:
        The text with ``[label](url)`` rewritten to ``label``.
    """
    return MD_LINK_RE.sub(r"\1", text)


def strip_urls(text: str) -> str:
    """Removes bare ``http(s)`` URLs from text.

    Args:
        text: The text to process.

    Returns:
        The text with URLs deleted.
    """
    return URL_RE.sub("", text)


def strip_bold_markers(text: str) -> str:
    """Removes markdown bold markers, keeping the emphasized text.

    Args:
        text: The text to process.

    Returns:
        The text with ``**bold**`` rewritten to ``bold``.
    """
    return BOLD_RE.sub(r"\1", text)


def remove_navigation_and_meta(text: str) -> str:
    """Drops site-navigation and promotional/metadata lines.

    Removes any line matching a ``META_PATTERNS`` entry (site name, "click
    here", "subscribe", copyright notices, etc.) or beginning with a
    navigation keyword such as "home" or "contact". Blank lines are kept.

    Args:
        text: The text to process.

    Returns:
        The text with navigation and metadata lines removed.
    """
    cleaned = []
    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            cleaned.append("")
            continue

        if any(p.search(line) for p in META_PATTERNS):
            continue

        if NAV_LINE_RE.match(stripped):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def remove_toc_blocks(text: str) -> str:
    """Removes paragraphs that look like a table of contents.

    A paragraph of more than four non-blank lines is dropped when more than
    40% of its lines match the table-of-contents line pattern (a numbered
    heading followed by dot leaders and a page number).

    Args:
        text: The text to process.

    Returns:
        The text with table-of-contents blocks removed.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    filtered = []

    for p in paragraphs:
        lines = [l for l in p.splitlines() if l.strip()]
        if len(lines) > 4:
            toc_like = sum(1 for l in lines if TOC_LINE_RE.match(l))
            if toc_like / len(lines) > 0.4:
                continue
        filtered.append(p)

    return "\n\n".join(filtered)


def separate_references(text: str) -> Tuple[str, Optional[str]]:
    """Splits text into main body and trailing references section.

    Once a paragraph whose first line is a recognized reference header
    (e.g., "References", "Bibliography") is seen, that paragraph and every
    subsequent paragraph are treated as references.

    Args:
        text: The text to process.

    Returns:
        A ``(main_text, ref_text)`` tuple. ``ref_text`` is None when no
        references section was found.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    main = []
    refs = []
    in_refs = False

    for p in paragraphs:
        first = p.strip().splitlines()[0].strip().lower() if p.strip() else ""

        if first in REFERENCE_HEADERS:
            in_refs = True

        if in_refs:
            refs.append(p)
        else:
            main.append(p)

    main_text = "\n\n".join(main).strip()
    ref_text = "\n\n".join(refs).strip() if refs else None

    return main_text, ref_text


def cleanup_whitespace(text: str) -> str:
    """Tidies trailing, inline, and vertical whitespace.

    Strips trailing whitespace from each line, collapses runs of spaces or
    tabs to a single space, collapses three or more newlines to two, and
    ensures the result ends with exactly one trailing newline.

    Args:
        text: The text to clean.

    Returns:
        The whitespace-normalized text.
    """
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


# ----------------------------
# RAG Cleaning Entry Point
# ----------------------------


def clean_text_for_rag(text: str, keep_references: bool = False) -> str:
    """Cleans scraped source text for RAG ingestion.

    Preserves paragraph structure while removing UI/promotional/navigation
    junk, table-of-contents blocks, markdown markup, and URLs, and
    separating a trailing references section from the main body.

    Args:
        text: The raw source text to clean.
        keep_references: Accepted for API compatibility; the returned
            text is the main body only, so references are dropped
            regardless of this flag.

    Returns:
        The cleaned main body text.
    """
    original_len = len(text)

    text = normalize_newlines(text)

    text = replace_markdown_links(text)
    text = strip_bold_markers(text)
    text = strip_urls(text)

    text = remove_navigation_and_meta(text)
    text = remove_toc_blocks(text)

    main_text, references = separate_references(text)
    main_text = cleanup_whitespace(main_text)

    # use the below if we want full-featured info
    # return {
    #    "clean_text": main_text,
    #    "references": references if keep_references else None,
    #    "stats": {
    #        "original_chars": original_len,
    #        "clean_chars": len(main_text),
    #        "retained_pct": round(len(main_text) / original_len * 100, 1)
    #            if original_len else 0.0
    #    }
    # }

    return main_text


# ----------------------------
# Rag entity extraction functions
# ----------------------------

# spaCy entity → simplified category mapping
LABEL_MAP = {
    "PERSON": "persons",
    "ORG": "organizations",
    "GPE": "locations",
    "LOC": "locations",
    "WORK_OF_ART": "works",
    "EVENT": "events",
    "DATE": "dates",
}

# labels we usually ignore
IGNORE_LABELS = {"CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "TIME", "MONEY"}

# allow some single token historical figures
PERSON_WHITELIST = {"Hitler", "Nixon", "Stalin", "Lenin", "JFK", "MLK"}

BLACKLIST = {"Darth Vader"}

# this spacy object used for entity classification for prompts
# "en_core_web_trf" has more accuracy but "en_core_web_sm" should be sufficient
nlp = spacy.load("en_core_web_sm")


def clean_entity(text: str) -> str:
    """Normalizes an entity surface form.

    Strips surrounding whitespace, removes a trailing possessive ``'s``,
    and collapses internal runs of whitespace to a single space.

    Args:
        text: The raw entity text as produced by spaCy.

    Returns:
        The cleaned entity string.
    """
    text = text.strip()

    # remove possessives
    text = re.sub(r"[’']s$", "", text)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text


def is_strong_person(name: str) -> bool:
    """Decides whether a candidate person name is reliable enough to keep.

    Single-token names are usually noise, so a name qualifies only if it is
    on ``PERSON_WHITELIST`` (well-known single-token historical figures) or
    consists of at least two tokens.

    Args:
        name: The candidate person name.

    Returns:
        True if the name should be retained, False otherwise.
    """
    if name in PERSON_WHITELIST:
        return True

    # require at least 2 tokens
    return len(name.split()) >= 2


def canonicalize_persons(persons: List[str]) -> List[str]:
    """Collapses person-name variants that share a last name.

    Groups names by their final token and keeps the longest variant in each
    group (preferring "John F. Kennedy" over "Kennedy").

    Args:
        persons: A list of person-name strings.

    Returns:
        A sorted list of canonical person names, one per distinct last name.
    """
    canonical = {}

    for p in persons:
        parts = p.split()

        last = parts[-1]

        # prefer longest name
        if last not in canonical or len(p) > len(canonical[last]):
            canonical[last] = p

    return sorted(canonical.values())


def canonicalize_orgs(orgs: List[str]) -> List[str]:
    """Collapses organization-name variants.

    Strips a leading "the ", deduplicates case-insensitively, and keeps the
    longest surface form for each distinct organization.

    Args:
        orgs: A list of organization-name strings.

    Returns:
        A sorted list of canonical organization names.
    """
    canonical = {}

    for o in orgs:
        o = re.sub(r"^the\s+", "", o, flags=re.I)

        key = o.lower()

        if key not in canonical or len(o) > len(canonical[key]):
            canonical[key] = o

    return sorted(set(canonical.values()))


def filter_dates(dates: List[str]) -> List[str]:
    """Keeps only date strings that begin with a four-digit year.

    Args:
        dates: A list of raw date strings from entity extraction.

    Returns:
        A sorted, deduplicated list of year-bearing date strings.
    """
    keep = []

    for d in dates:

        # keep real years
        if re.match(r"\b\d{4}\b", d):
            keep.append(d)

    return sorted(set(keep))


def limit_entities(items: List[str], limit: int) -> List[str]:
    """Truncates an entity list to at most ``limit`` items.

    Args:
        items: The entity list to cap.
        limit: The maximum number of items to retain.

    Returns:
        A list containing at most ``limit`` leading items from ``items``.
    """
    return items[:limit]


def extract_entities(text: str) -> List[Dict[str, str]]:
    """Runs the spaCy NER pipeline over text.

    Args:
        text: The text to analyze.

    Returns:
        A list of dicts, each with ``text`` (the entity surface form) and
        ``label`` (the spaCy entity label) keys.
    """
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        entities.append({"text": ent.text, "label": ent.label_})

    return entities


def group_entities(entities: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """Buckets, cleans, canonicalizes, and caps extracted entities.

    Maps spaCy labels onto simplified categories via ``LABEL_MAP``, dropping
    ignored labels and blacklisted surface forms. Persons are filtered for
    strength and canonicalized; organizations are canonicalized; locations,
    works, and events are deduplicated; dates are filtered to years. Each
    category is capped (persons 5, organizations 4, locations 3) and empty
    categories are removed.

    Args:
        entities: Entity dicts as returned by ``extract_entities``.

    Returns:
        A dict mapping non-empty category names to their entity lists.
    """
    buckets = {
        "persons": [],
        "organizations": [],
        "locations": [],
        "works": [],
        "events": [],
        "dates": [],
    }

    for ent in entities:

        label = ent["label"]

        if label in IGNORE_LABELS:
            continue

        category = LABEL_MAP.get(label)

        if not category:
            continue

        text = clean_entity(ent["text"])

        if text in BLACKLIST:
            continue

        buckets[category].append(text)

    # --- persons ---
    persons = [p for p in buckets["persons"] if is_strong_person(p)]
    persons = canonicalize_persons(persons)

    # --- organizations ---
    orgs = canonicalize_orgs(buckets["organizations"])

    # --- locations ---
    locations = sorted(set(buckets["locations"]))

    # --- works ---
    works = sorted(set(buckets["works"]))

    # --- events ---
    events = sorted(set(buckets["events"]))

    # --- dates ---
    dates = filter_dates(buckets["dates"])

    result = {
        "persons": limit_entities(persons, 5),
        "organizations": limit_entities(orgs, 4),
        "locations": limit_entities(locations, 3),
        "works": works,
        "events": events,
        "dates": dates,
    }

    # remove empty categories
    return {k: v for k, v in result.items() if v}


def extract_and_group_entities(text: str) -> Dict[str, List[str]]:
    """Extracts entities from text and returns them grouped by category.

    Convenience wrapper composing ``extract_entities`` and ``group_entities``.

    Args:
        text: The text to analyze.

    Returns:
        A dict mapping non-empty category names to their entity lists.
    """
    return group_entities(extract_entities(text))


def extract_query_entities(text: str) -> Dict[str, List[str]]:
    """Extracts the most salient entities from a search query.

    Groups query entities, then scores each by an additive blend of position
    (earlier in the query scores higher), length/specificity, and a
    per-category type weight from ``ENTITY_WEIGHTS``. Only the top-scoring
    entities are kept, with much smaller per-category caps than document
    extraction uses (persons 3, organizations/locations 2, otherwise 1).

    Args:
        text: The query text.

    Returns:
        A dict mapping non-empty category names to their most salient
        entity lists.
    """
    raw_entities = extract_entities(text)
    grouped = group_entities(raw_entities)

    salient = {}

    for category, items in grouped.items():

        scored = []

        for ent in items:

            # --- position score (earlier = more important) ---
            pos = text.find(ent)
            position_score = 1.0 - (pos / max(len(text), 1))

            # --- length / specificity ---
            length_score = len(ent.split()) * 0.2

            # --- type weight ---
            type_weight = ENTITY_WEIGHTS.get(category, 0.5)

            score = position_score + length_score + type_weight

            scored.append((score, ent))

        scored.sort(reverse=True)

        # much smaller caps for queries
        top_n = {"persons": 3, "organizations": 2, "locations": 2}.get(
            category, 1
        )

        salient[category] = [ent for _, ent in scored[:top_n]]

    return {k: v for k, v in salient.items() if v}


# ------------------ RERANKING ------------------

ENTITY_WEIGHTS = {
    "persons": 1.0,
    "organizations": 0.8,
    "events": 1.0,
    "locations": 0.5,
}


def entity_overlap_score(
    query_set: set, chunk_entities: Dict[str, List[str]]
) -> float:
    """Scores a chunk by the fraction of query entities it contains.

    Flattens all of the chunk's entities into a single set and divides the
    size of its intersection with the query set by the query-set size, so
    long chunks cannot dominate purely by having more entities.

    Args:
        query_set: The set of entity strings from the query.
        chunk_entities: The chunk's grouped entities (category -> list).

    Returns:
        The normalized overlap in [0, 1]; 0.0 when there is no overlap.
    """
    chunk_set = set(e for v in chunk_entities.values() for e in v)

    overlap = query_set & chunk_set

    if not overlap:
        return 0.0

    # normalized overlap (prevents long chunks dominating)
    return len(overlap) / len(query_set)


def boost_score(bm25_score: float, entity_score: float) -> float:
    """Adds an entity-overlap boost on top of a BM25 score.

    Args:
        bm25_score: The chunk's base BM25 relevance score.
        entity_score: A normalized entity-overlap score in [0, 1].

    Returns:
        The BM25 score plus the entity score scaled by a fixed maximum
        boost of 0.5.
    """
    MAX_BOOST = 0.5

    return bm25_score + (entity_score * MAX_BOOST)


def passes_entity_gate(
    query_entities: Dict[str, List[str]],
    chunk_entities: Dict[str, List[str]],
) -> bool:
    """Checks whether a chunk shares at least one entity with the query.

    Used as a hard filter before scoring: a chunk that matches no query
    entity in any category is dropped from reranking.

    Args:
        query_entities: The query's grouped entities (category -> list).
        chunk_entities: The chunk's grouped entities (category -> list).

    Returns:
        True if any query entity appears in the chunk's same-category set.
    """
    for category, q_entities in query_entities.items():
        chunk_set = set(chunk_entities.get(category, []))

        for ent in q_entities:
            if ent in chunk_set:
                return True

    return False


def weighted_entity_score(
    query_entities: Dict[str, List[str]],
    chunk_entities: Dict[str, List[str]],
) -> float:
    """Scores entity overlap with per-category weighting.

    Each query entity contributes its category weight (from
    ``ENTITY_WEIGHTS``, default 0.5) to the maximum possible score, and
    contributes the same weight to the actual score when it appears in the
    chunk's same-category set. The result is normalized to [0, 1].

    Args:
        query_entities: The query's grouped entities (category -> list).
        chunk_entities: The chunk's grouped entities (category -> list).

    Returns:
        The weighted, normalized overlap in [0, 1]; 0.0 when the query has
        no weighable entities.
    """
    score = 0.0
    max_possible = 0.0

    for category, q_entities in query_entities.items():

        weight = ENTITY_WEIGHTS.get(category, 0.5)

        chunk_set = set(chunk_entities.get(category, []))

        for ent in q_entities:
            max_possible += weight

            if ent in chunk_set:
                score += weight

    if max_possible == 0:
        return 0.0

    return score / max_possible  # normalize to [0,1]


def entity_frequency_score(
    query_entities: Dict[str, List[str]], chunk: Dict
) -> float:
    """Scores a chunk by how often its query entities recur in its text.

    Higher entity frequency is treated as more important: "Roosevelt
    mentioned once" scores lower than "Roosevelt discussed throughout the
    chunk". Occurrence counts are dampened with a logarithm so frequent
    entities don't dominate everything, and each entity's contribution is
    scaled by its category weight from ``ENTITY_WEIGHTS``.

    Args:
        query_entities: The query's grouped entities (category -> list).
        chunk: A chunk dict with a ``text`` key and an
            ``entities_grouped`` key.

    Returns:
        The accumulated, log-dampened, weighted frequency score.
    """

    text = chunk["text"]
    chunk_entities = chunk.get("entities_grouped", {})

    score = 0.0

    for category, q_entities in query_entities.items():

        weight = ENTITY_WEIGHTS.get(category, 0.5)

        for ent in q_entities:

            if ent in chunk_entities.get(category, []):

                # count occurrences
                freq = text.count(ent)

                # dampen with log (prevents spam boosting)
                score += weight * math.log(1 + freq)

    return score


def split_sentences(text: str) -> List[str]:
    """Splits text into sentences on runs of ``.``, ``!``, or ``?``.

    Args:
        text: The text to split.

    Returns:
        A list of sentence fragments.
    """
    return re.split(r"[.!?]+", text)


def entity_proximity_score(
    query_entities: Dict[str, List[str]], chunk: Dict
) -> float:
    """Scores how often query entities co-occur within the same sentence.

    Detects whether entities are actually connected rather than merely
    co-present in the chunk: each sentence containing two or more distinct
    query entities adds 1.0 to the score. Returns 0.0 when the query has
    fewer than two entities, since no proximity signal is then possible.

    Args:
        query_entities: The query's grouped entities (category -> list).
        chunk: A chunk dict with a ``text`` key.

    Returns:
        The count of sentences containing at least two query entities.
    """
    text = chunk["text"]
    sentences = split_sentences(text)

    flat_query = [ent for v in query_entities.values() for ent in v]

    if len(flat_query) < 2:
        return 0.0  # no proximity signal possible

    score = 0.0

    for sentence in sentences:

        matches = [ent for ent in flat_query if ent in sentence]

        if len(matches) >= 2:
            # strong signal: multiple entities in same sentence
            score += 1.0

    return score


def rerank_chunks(query: str, chunks: List[Dict]) -> List[Dict]:
    """Reranks retrieved chunks by entity relevance to the query.

    Extracts salient query entities, drops any chunk that fails the entity
    gate (shares no entity with the query), scores the survivors with
    ``score_chunk``, and returns them sorted by descending score.

    Args:
        query: The search query text.
        chunks: A list of chunk dicts, each with an ``entities_grouped``
            key and the fields required by ``score_chunk``.

    Returns:
        The gate-passing chunks ordered from highest to lowest score.
    """
    query_entities = extract_query_entities(query)

    reranked = []

    for chunk in chunks:

        # --- gate ---
        if not passes_entity_gate(query_entities, chunk["entities_grouped"]):
            continue

        score = score_chunk(None, query_entities, chunk)

        reranked.append((score, chunk))

    reranked.sort(reverse=True)

    return [c for _, c in reranked]


def score_chunk(
    query_vec, query_entities: Dict[str, List[str]], chunk: Dict
) -> float:
    """Computes a chunk's combined rerank score.

    Blends the chunk's base BM25 score with three entity signals:
    weighted entity overlap, log-dampened entity frequency, and
    same-sentence entity proximity, using fixed tunable weights.

    Args:
        query_vec: The query embedding vector. Currently unused; reserved
            for a future vector-similarity term.
        query_entities: The query's grouped entities (category -> list).
        chunk: A chunk dict with ``bm25_score``, ``text``, and
            ``entities_grouped`` keys.

    Returns:
        The combined relevance score.
    """
    bm25 = chunk["bm25_score"]

    entity_overlap = weighted_entity_score(
        query_entities, chunk["entities_grouped"]
    )

    freq_score = entity_frequency_score(query_entities, chunk)

    proximity_score = entity_proximity_score(query_entities, chunk)

    # weights (tune these)
    return (
        bm25 + 0.4 * entity_overlap + 0.3 * freq_score + 0.5 * proximity_score
    )
