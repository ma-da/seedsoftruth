import re
from typing import Dict, Optional, List
import spacy
import logging_config
import math

rag_cleaner_logger = logging_config.get_logger("rag_cleaner")

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

NAV_LINE_RE = re.compile(r"^\s*(home|about|contact|privacy|terms)\b", re.IGNORECASE)


# ----------------------------
# Core utilities
# ----------------------------

def normalize_newlines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text


def replace_markdown_links(text: str) -> str:
    return MD_LINK_RE.sub(r"\1", text)


def strip_urls(text: str) -> str:
    return URL_RE.sub("", text)


def strip_bold_markers(text: str) -> str:
    return BOLD_RE.sub(r"\1", text)


def remove_navigation_and_meta(text: str) -> str:
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


def separate_references(text: str) -> (str, Optional[str]):
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
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


# ----------------------------
# RAG Cleaning Entry Point
# ----------------------------

def clean_text_for_rag(
    text: str,
    keep_references: bool = False
) -> str:
    paragraphs = re.split(r"\n\s*\n", text)
    """
    RAG-oriented cleaner:
    - Preserves paragraph structure
    - Removes UI / promo / navigation junk
    - Removes TOC blocks
    - Optionally separates references
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
    #return {
    #    "clean_text": main_text,
    #    "references": references if keep_references else None,
    #    "stats": {
    #        "original_chars": original_len,
    #        "clean_chars": len(main_text),
    #        "retained_pct": round(len(main_text) / original_len * 100, 1)
    #            if original_len else 0.0
    #    }
    #}

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
    "DATE": "dates"
}

# labels we usually ignore
IGNORE_LABELS = {
    "CARDINAL",
    "ORDINAL",
    "QUANTITY",
    "PERCENT",
    "TIME",
    "MONEY"
}

# allow some single token historical figures
PERSON_WHITELIST = {
    "Hitler",
    "Nixon",
    "Stalin",
    "Lenin",
    "JFK",
    "MLK"
}

BLACKLIST = {"Darth Vader"}

# this spacy object used for entity classification for prompts
# "en_core_web_trf" has more accuracy but "en_core_web_sm" should be sufficient
nlp = spacy.load("en_core_web_sm")

def clean_entity(text):
    text = text.strip()

    # remove possessives
    text = re.sub(r"[’']s$", "", text)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text

def is_strong_person(name):

    if name in PERSON_WHITELIST:
        return True

    # require at least 2 tokens
    return len(name.split()) >= 2


def canonicalize_persons(persons):

    canonical = {}

    for p in persons:
        parts = p.split()

        last = parts[-1]

        # prefer longest name
        if last not in canonical or len(p) > len(canonical[last]):
            canonical[last] = p

    return sorted(canonical.values())


def canonicalize_orgs(orgs):

    canonical = {}

    for o in orgs:
        o = re.sub(r"^the\s+", "", o, flags=re.I)

        key = o.lower()

        if key not in canonical or len(o) > len(canonical[key]):
            canonical[key] = o

    return sorted(set(canonical.values()))


def filter_dates(dates):

    keep = []

    for d in dates:

        # keep real years
        if re.match(r"\b\d{4}\b", d):
            keep.append(d)

    return sorted(set(keep))


def limit_entities(items, limit):

    return items[:limit]


def extract_entities(text):
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_
        })

    return entities

def group_entities(entities):

    buckets = {
        "persons": [],
        "organizations": [],
        "locations": [],
        "works": [],
        "events": [],
        "dates": []
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
        "dates": dates
    }

    # remove empty categories
    return {k: v for k, v in result.items() if v}


def extract_and_group_entities(text):
    return group_entities(extract_entities(text))

def extract_query_entities(text):

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
            type_weight = ENTITY_TYPE_WEIGHTS.get(category, 0.5)

            score = position_score + length_score + type_weight

            scored.append((score, ent))

        scored.sort(reverse=True)

        # much smaller caps for queries
        top_n = {
            "persons": 3,
            "organizations": 2,
            "locations": 2
        }.get(category, 1)

        salient[category] = [ent for _, ent in scored[:top_n]]

    return {k: v for k, v in salient.items() if v}

# ------------------ RERANKING ------------------

ENTITY_TYPE_WEIGHTS = {
    "persons": 1.0,
    "events": 1.0,
    "organizations": 0.7,
    "locations": 0.4,
    "works": 0.5,
    "dates": 0.2
}


def weighted_entity_overlap(query_entities, chunk_entities):
    score = 0.0

    for category, q_entities in query_entities.items():
        weight = ENTITY_TYPE_WEIGHTS.get(category, 0.5)
        chunk_set = set(chunk_entities.get(category, []))

        for ent in q_entities:
            if ent in chunk_set:
                score += weight

    return score


def entity_frequency_score(query_entities, chunk_text):
    score = 0.0

    for category, q_entities in query_entities.items():
        weight = ENTITY_TYPE_WEIGHTS.get(category, 0.5)
        text_lower = chunk_text.lower()

        for ent in q_entities:
            ent_lower = ent.lower()
            freq = text_lower.count(ent_lower)
            if freq > 0:
                score += weight * math.log(1 + freq)

    return min(score, 2.0)  # cap


def split_sentences(text):
    return re.split(r"[.!?]+", text)


def entity_proximity_score(query_entities, chunk_text):
    sentences = split_sentences(chunk_text)

    flat_query = [
        ent for v in query_entities.values() for ent in v
    ]

    if len(flat_query) < 2:
        return 0.0

    score = 0.0

    for sentence in sentences:
        sentence_lower = sentence.lower()
        matches = [ent for ent in flat_query if ent in sentence_lower]

        if len(matches) >= 2:
            score += 1.0

    return min(score, 2.0)  # cap


def passes_entity_gate(query_entities, chunk_entities):
    for category, q_entities in query_entities.items():
        chunk_set = set(chunk_entities.get(category, []))
        for ent in q_entities:
            if ent in chunk_set:
                return True
    return False


def rerank_chunks(query, chunks, query_entities):
    reranked = []
    filtered = []

    # first pass filters out low-value chunks
    for chunk in chunks:
        chunk_entities = chunk.get("entities_grouped", {})
        if not query_entities or passes_entity_gate(query_entities, chunk_entities):
            filtered.append(chunk)

    # fallback if too strict
    if not filtered:
        filtered = chunks
        rag_cleaner_logger.info(f"rerank_chunks filtered fallback to default, size {len(filtered)}")
    else:
        num_filtered = len(chunks) - len(filtered)
        rag_cleaner_logger.info(f"rerank_chunks filtered out {num_filtered} chunks, curr size {len(filtered)}")

    for chunk in filtered:
        chunk_entities = chunk.get("entities_grouped", {})
        text = chunk.get("text", "")
        text_lower = text.lower()

        bm25 = chunk.get("score_bm25", 0.0)
        overlap = weighted_entity_overlap(query_entities, chunk_entities)
        freq = entity_frequency_score(query_entities, text_lower)
        proximity = entity_proximity_score(query_entities, text_lower)

        final_score = (
            bm25
            + 0.4 * overlap
            + 0.3 * freq
            + 0.5 * proximity
        )

        rag_cleaner_logger.info(f"rerank chunk: final_score {final_score}, bm25 {bm25}, overlap {overlap}, freq {freq}, proximity {proximity}")
        reranked.append((final_score, chunk))

    reranked.sort(reverse=True)

    # log top final scores
    top_scores = ""
    for chunk_info in reranked[:5]:
        top_scores += str(chunk_info[0]) + ", "
    rag_cleaner_logger.info(f"final reranked chunks: {top_scores}")

    return [c for _, c in reranked]