# seedsoftruth/tools

Corpus prep, cleaning, and hybrid-FTS-database build tools for the Seeds of Truth RAG app.

These tools form a pipeline. Each one is also runnable on its own. They
are organized into subdirectories by pipeline stage:

```
tools/
├── ingest/   raw documents → text   (EPUB→PDF, PDF→text, MD→text)
├── clean/    clean & filter corpus text
├── index/    build the hybrid-FTS retrieval database
└── eval/     analyze chunks & probe retrieval quality
```

```
   .epub  ──►  .pdf   ──►  .txt   ──►  filtered .txt  ──►  cleaned .txt  ──►  SQLite hybrid-FTS DB
            epub_to_pdf_batch.py     mineru_batch.sh    filter_corpus.py   clean_web_corpus.py   corpus_to_hybrid_db.py
            (Calibre)                (MinerU)           (chardet/ftfy/bs4) (block-level cleaner) (chunks + FTS + NER + optional topics)

   .md     ──►  .txt
            convert_md_to_txt.sh
            (pandoc)

   transcripts (Speaker N HH:MM:SS) ──► dialogue only
            clean_transcript.py
```

You don't have to use every step — pick the ones that match your input.

## File map

### `ingest/` — raw documents to text

| File | Type | Purpose |
| --- | --- | --- |
| `epub_to_pdf_batch.py` | CLI | Calibre wrapper: batch convert `.epub` → `.pdf`. |
| `mineru_batch.sh` | shell | MinerU CLI wrapper: batch convert `.pdf` → text output. |
| `convert_md_to_txt.sh` | shell | Pandoc wrapper: every `.md` in a tree → sibling `.txt`. |

### `clean/` — clean & filter corpus text

| File | Type | Purpose |
| --- | --- | --- |
| `filter_corpus.py` | CLI | File-level quality gate: rejects garbled / non-natural-language files (chardet + ftfy + sliding-window NL detection). |
| `clean_web_corpus.py` | CLI | Block-level cleaner for a directory of `.txt` files (markdown, URLs, TOC/index, refs, meta lines, paragraph rejoin). Outputs `_cleaned.txt` files. |
| `clean_chunks.py` | CLI | Sentence-pattern filter that runs over an existing DB to drop chrome/CTA/cookie banners from `chunks.fulltext_text` and resync FTS. Also imported by `index/corpus_to_hybrid_db.py` during ingest. |
| `clean_transcript.py` | CLI | Drops `Speaker N HH:MM:SS` lines from podcast / video transcripts. |
| `text_cleaners.py` | library | Shared cleaning helpers imported by the CLIs in this directory. Single source of truth for patterns / heuristics. |

### `index/` — build the retrieval database

| File | Type | Purpose |
| --- | --- | --- |
| `corpus_to_hybrid_db.py` | CLI | Chunks a corpus into a SQLite DB matching `data/gamma_master_hybrid_fts_stage3.db`. Optional spaCy NER, optional BGE+Llama topic classification. Two subcommands: `ingest`, `enrich`. Delegates sentence cleaning to `clean/clean_chunks.py`. |

### `eval/` — analyze & probe

| File | Type | Purpose |
| --- | --- | --- |
| `analyze_chunks.py` | CLI | Inspect chunk-length and other distributions in a hybrid-FTS DB. |
| `prober.py` | CLI | Ad-hoc probes against the live retrieval pipeline. |
| `trineday-rag.ipynb` | notebook | Exploratory RAG notebook. |

## Recommended end-to-end pipeline

Assume you have a folder of EPUBs at `/data/raw_epubs` and want a SQLite hybrid-FTS DB at `/data/my_corpus.db` with subset name `"My Source"`.

```bash
# 1) EPUB -> PDF
python tools/ingest/epub_to_pdf_batch.py /data/raw_epubs /data/pdfs

# 2) PDF -> text (MinerU)
tools/ingest/mineru_batch.sh /data/pdfs /data/mineru_out

# 3) If MinerU produced .md, convert -> .txt
tools/ingest/convert_md_to_txt.sh /data/mineru_out

# 4) Collect every .txt under /data/mineru_out into one flat dir
#    (do this however you like — find + rsync, a python loop, etc.)
mkdir -p /data/all_txt
find /data/mineru_out -name '*.txt' -exec cp {} /data/all_txt/ \;

# 5) File-level quality gate. Garbled/non-NL files get moved to /data/rejected.
python tools/clean/filter_corpus.py \
    --input-dir  /data/all_txt \
    --output-dir /data/filtered_txt \
    --reject-dir /data/rejected

# 6) Block-level cleaner. Drops markdown chrome, URLs, TOC, index, refs.
python tools/clean/clean_web_corpus.py \
    --input-dir  /data/filtered_txt \
    --output-dir /data/cleaned_txt \
    --strip_pre 5 --strip_post 5

# 7) Build the hybrid-FTS DB. Chunks + FTS5 indexes + spaCy NER by default.
#    Note: corpus_to_hybrid_db.py also applies the sentence-pattern filter from
#    clean_chunks.py to each file's text right before chunking, so even if you
#    skip step 6 the output won't be full of cookie banners and "click here".
python tools/index/corpus_to_hybrid_db.py ingest \
    --corpus      /data/cleaned_txt \
    --db          /data/my_corpus.db \
    --subset-name "My Source"

# 8) (Optional) Run topic classification later, when GPU is available.
python tools/index/corpus_to_hybrid_db.py enrich \
    --db /data/my_corpus.db --topics
```

## Per-tool quick reference

### `corpus_to_hybrid_db.py`

Two subcommands. See `--help` for each.

```bash
# Ingest a corpus
python tools/index/corpus_to_hybrid_db.py ingest \
    --corpus /data/cleaned_txt \
    --db     /data/my_corpus.db \
    --subset-name "My Source" \
    [--domain intelligence_security] \
    [--url-map /data/url_map.csv] \
    [--chunk-words 480] [--overlap-words 80] [--min-words 80] \
    [--no-dedupe] [--no-clean] [--no-ner] \
    [--with-topics] \
    [--spacy-model en_core_web_sm] \
    [--clean-existing]

# Enrich an existing DB
python tools/index/corpus_to_hybrid_db.py enrich \
    --db /data/my_corpus.db \
    [--entities] [--topics] [--rebuild]
```

The URL map CSV has two columns: `filename,source_url`. Files not in the map get an empty source_url.

### `clean_chunks.py`

DRY-RUN by default. Backs up the DB (timestamped `.bak.<ts>`) before writing unless `--no-backup`.

```bash
# Preview
python tools/clean/clean_chunks.py --db data/gamma_master_hybrid_fts_stage3.db
# Apply
python tools/clean/clean_chunks.py --db data/gamma_master_hybrid_fts_stage3.db --apply
# FTS resync only (after an external script rewrote fulltext_text)
python tools/clean/clean_chunks.py --db data/gamma_master_hybrid_fts_stage3.db --apply --fts-only
```

### `clean_web_corpus.py`

Block-level cleaner. Output files get `_cleaned` appended to the stem. Re-running the tool over its output is safe (already-cleaned files are skipped).

```bash
python tools/clean/clean_web_corpus.py \
    --input-dir  /data/filtered_txt \
    --output-dir /data/cleaned_txt \
    [--dry-run] \
    [--strip_pre 0]  [--strip_post 0]     \  # each 0–25 (percent)
    [--cut-marker "**What you can do:**"] \  # pass "" to disable
    [--min-block-chars 220]               \
    [--recursive]
```

### `filter_corpus.py`

File-level quality gate. Files that pass go to `--output-dir` as `.txt`. Files that fail get moved (or copied with `--copy-rejects`) to `--reject-dir`. A CSV decision log is written to `<output-dir>/clean_log.csv` by default.

```bash
python tools/clean/filter_corpus.py \
    --input-dir  /data/all_txt \
    --output-dir /data/filtered_txt \
    --reject-dir /data/rejected \
    [--log /data/clean_log.csv]      \
    [--dry-run]                       \
    [--copy-rejects]                  \
    [--window-size 60]                \
    [--alpha-min 0.80]                \
    [--max-nonascii 0.20]             \
    [--garbled-min-chars 400]         \
    [--garbled-alpha-frac 0.55]

# Requires: pip install chardet beautifulsoup4 lxml ftfy
```

**Note on output format:** The original `preprocessing/corpus_filtering.py` wrote
the aggressively-normalized (punctuation-stripped, single-line) text to disk,
which made the output unusable for downstream cleaners or chunkers. This port
writes the encoding-fixed + HTML-stripped version *with paragraph structure
intact*, and uses the destructive normalization only for the keep/reject
decision. The kept output is suitable as input to `clean_web_corpus.py` or
`corpus_to_hybrid_db.py ingest` directly.

### `clean_transcript.py`

Drops `Speaker N HH:MM:SS` lines from transcripts.

```bash
# Single file
python tools/clean/clean_transcript.py path/to/raw.txt path/to/clean.txt

# Batch
python tools/clean/clean_transcript.py \
    --input-dir  /data/raw_transcripts \
    --output-dir /data/clean_transcripts
```

### `epub_to_pdf_batch.py`

```bash
python tools/ingest/epub_to_pdf_batch.py /data/epubs /data/pdfs
# Requires Calibre's ebook-convert on PATH:
#   macOS:   brew install --cask calibre
#   Ubuntu:  sudo apt install calibre
```

### `mineru_batch.sh`

```bash
tools/ingest/mineru_batch.sh /data/pdfs /data/mineru_out
# Requires:  pip install mineru
# Failures logged to:  /data/mineru_out/mineru_failures.log
```

### `convert_md_to_txt.sh`

```bash
tools/ingest/convert_md_to_txt.sh /data/some_tree_with_md_files
# Requires pandoc on PATH:
#   macOS:   brew install pandoc
#   Ubuntu:  sudo apt install pandoc
```

### `text_cleaners.py` (library, not a CLI)

Shared helpers. Import these in your own scripts rather than copy-pasting regex:

```python
from text_cleaners import (
    # block-level
    clean_text_web_corpus, split_blocks, strip_blocks,
    is_toc_block, is_index_block, split_reference_blocks,

    # individual passes
    normalize_newlines, cut_after_marker,
    replace_markdown_links, strip_bold_markers, strip_urls,
    remove_meta_lines, cleanup_whitespace, normalize_paragraph_linebreaks,

    # transcript
    is_speaker_line, strip_speaker_lines,

    # file-level quality gate
    read_text_with_detection, strip_html, normalize_and_clean,
    has_natural_language_run, is_garbled,
)
```

## Dependencies summary

| Tool | Python deps | System deps |
| --- | --- | --- |
| `corpus_to_hybrid_db.py` (basic) | stdlib only | — |
| `corpus_to_hybrid_db.py` (with NER) | `spacy` + a spaCy model (`en_core_web_sm` by default) | — |
| `corpus_to_hybrid_db.py` (with topics) | `torch`, `transformers`, `bitsandbytes`, `sentence-transformers` | CUDA-capable GPU recommended |
| `clean_chunks.py` | stdlib only | — |
| `clean_web_corpus.py` | stdlib only | — |
| `clean_transcript.py` | stdlib only | — |
| `filter_corpus.py` | `chardet`, `beautifulsoup4`, `lxml`, `ftfy` | — |
| `epub_to_pdf_batch.py` | stdlib only | Calibre (`ebook-convert`) |
| `mineru_batch.sh` | — | MinerU (`mineru` CLI) |
| `convert_md_to_txt.sh` | — | `pandoc` |

Install everything in one shot:

```bash
pip install spacy chardet beautifulsoup4 lxml ftfy
python -m spacy download en_core_web_sm

# Optional, only if you'll run --with-topics:
pip install torch transformers bitsandbytes sentence-transformers
```

## Notes on cleaning granularity

The repo has cleaning at three different granularities. Use the right one for your needs:

1. **File-level (`filter_corpus.py`)** — Is this entire file even worth keeping? Quality gate; rejects garbage outright. Use once, at the start.
2. **Block-level (`clean_web_corpus.py`)** — Within a file, drop entire paragraph blocks that look like TOC, index, refs, or that are short / SCREAMING / promotional.
3. **Sentence-level (`clean_chunks.py`, also wired into `corpus_to_hybrid_db.py ingest`)** — Within prose, drop individual sentences that match navigation chrome / CTA / cookie banner / "click here" patterns. Doesn't affect surrounding sentences.

The three are complementary and applying them in that order (file → block → sentence) is the cleanest path. Skipping the earlier ones doesn't break anything; later passes are designed to be robust to whatever made it through.
