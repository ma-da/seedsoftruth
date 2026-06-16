# Stage 2 — Cleaning the Corpus

The crawler in stage 1 saved one `.txt` file per page. But a lot of
that text is junk you don't want to expose to retrieval: navigation
menus, "subscribe" CTAs, repeated article footers, listing pages whose
"content" is a concatenation of other articles you already have. This
stage runs `tools/clean/clean_web_corpus.py` to filter and clean the corpus
before ingest.

## Why cleaning matters for RAG

Two failure modes that motivated the current cleaner:

- **Boilerplate poisons retrieval.** If every article in a subset ends
  with the same 4-paragraph "about us" closer, then a query for *any*
  word in that closer will match every article. The closer becomes a
  super-common term across the corpus, BM25 inflates its IDF, and the
  ranker thinks every article is equally relevant. By cutting the
  shared footer once, every article's BM25 score reflects its actual
  topical content.
- **Listing pages create near-duplicates.** Substack's `/archive` and
  `/t/<topic>` pages are concatenations of article titles + leading
  sentences. If you ingest those alongside the real articles, the
  retriever surfaces the listing page (which has *all* the keywords) as
  the top result for half your queries, and the user sees a useless
  snippet instead of the actual article body.

The cleaner's defaults reflect these failure modes specifically.

## The tool

### `tools/clean/clean_web_corpus.py`

Reads every `.txt` under `--input-dir`, runs each file through the
pipeline below, and writes the result to `--output-dir/<name>_cleaned.txt`.

Pipeline (per file):

1. **File-level rejection** (new, opt-out-able). Drop the file
   entirely if any of:
   - It matches a `--drop-pattern` glob.
   - It's a JS-required Substack shell (matches one of several
     hard-coded signatures and is shorter than ~600 chars).
   - It's a Substack listing/archive page (carries the Substack nav
     strip + 5+ `<date> • <author>` snippets).
2. **Cut everything after a marker.** If `--cut-marker` (default
   `**What you can do:**`) appears in the file, truncate at that point.
   Used for WantToKnow-style articles whose "what you can do" section
   is action-CTA, not content.
3. **Cut repeated footer.** If any of `--repeated-footer-prefix`
   (defaults include the PEERS/WTK closer) appears, truncate from the
   earliest occurrence. This is the cut that protects retrieval from
   the boilerplate-poisoning failure above.
4. **Markdown normalize.** `[label](url)` → `label`; `**bold**` →
   `bold`. The Substack→TXT extraction sometimes leaves these.
5. **Strip bare URLs.**
6. **Strip promotional / meta lines.** Lines starting with `Note:`,
   `For more information`, or containing `WantToKnow.info`, `PEERS`,
   or `click here`.
7. **Normalize whitespace** + rejoin wrapped paragraph lines into
   single long-line paragraphs.
8. **Split into paragraph blocks.** Optionally drop the first
   `--strip_pre` percent and last `--strip_post` percent of blocks
   (front/back-matter stripping for PDF→TXT conversions).
9. **Drop TOC-like and book-index-like blocks** via heuristic patterns.
10. **Split off references.** If a block starts with "References",
    "Bibliography", etc., that block and everything after is kept
    separate (not written to the cleaned file by default).
11. **Drop tiny and SCREAMING-ALL-CAPS blocks.** `--min-block-chars`
    (default 220) catches headings and section banners.
12. **Post-clean size gate.** If the cleaned file is shorter than
    `--min-file-chars` (default 400), drop it.

A final summary at the end of the run shows kept/dropped/skipped
breakdown by reason — useful for spotting if your filters are too
aggressive or too lenient.

### `tools/clean/text_cleaners.py`

The shared helper module the CLIs import from. Don't add a new pattern
or rule in the CLI — add it here and the CLI inherits it. Notable
exports:

```python
DEFAULT_REPEATED_FOOTERS    # the PEERS/WTK closer prefixes
JS_REQUIRED_STUB_PATTERNS   # signatures for Substack JS shells
clean_text_web_corpus(...)  # the full per-file pipeline
evaluate_file_rejection(...) # file-level gate (used by the CLI)
cut_repeated_footer(...)
is_listing_page(...)
is_js_required_stub(...)
```

`text_cleaners.py` is also the source of truth for the file-level
natural-language detection used by `tools/clean/filter_corpus.py` (a
heavier-weight quality gate for OCR-derived PDFs).

### `tools/clean/clean_chunks.py`

A sister tool that operates at the **sentence** level inside each
chunk after the chunker has split the document. It catches navigation
chrome and CTA copy that survived `clean_web_corpus.py`. It's called
by default from `corpus_to_hybrid_db.py` (stage 3) — you don't usually
invoke it directly. Pass `--no-clean` to the ingest step to skip it.

## Common usage

The default flags are tuned for Substack + WTK-shaped corpora. For
those:

```bash
python3 tools/clean/clean_web_corpus.py \
  --input-dir webscraper/corpus1 \
  --output-dir webscraper/corpus1_cleaned
```

To restore the pre-update behavior (no file-level rejection, keep
PEERS footer):

```bash
python3 tools/clean/clean_web_corpus.py \
  --input-dir <in> --output-dir <out> --no-default-filters
```

To disable one specific filter:

```bash
# Keep listing pages
python3 tools/clean/clean_web_corpus.py --input-dir <in> --output-dir <out> \
  --no-drop-listing-pages
```

To drop additional files by pattern (e.g., all sub-pages of a
particular author):

```bash
python3 tools/clean/clean_web_corpus.py --input-dir <in> --output-dir <out> \
  --drop-pattern 'substack.com_@someuser*'
```

If your corpus uses a different repeated-footer than PEERS/WTK, override:

```bash
python3 tools/clean/clean_web_corpus.py --input-dir <in> --output-dir <out> \
  --repeated-footer-prefix "Copyright © 2026 Acme Publishing" \
  --repeated-footer-prefix "Subscribe to our daily digest"
```

`--repeated-footer-prefix` is repeatable; the earliest matching prefix
wins.

## How to tune the knobs

Run with `--dry-run` first. The output prints, per file:

```
[DRY-RUN] foo.txt → foo_cleaned.txt | retained: 78.4% (2195/2801 chars) |
                    blocks: 10→10 (strip_pre=0%, strip_post=0%) |
                    footer_cut: 989 chars
[DROP]    bar.txt → listing_page:bylines=24
```

Things to watch for:

- **Retention <50%** on real articles → probably `--min-block-chars`
  is too aggressive for this corpus. Try `--min-block-chars 80`.
- **Real articles getting dropped as `too_short`** → the cleaner ate
  too much. Same fix.
- **`listing_page` drops on real articles** → the `is_listing_page`
  heuristic is over-firing. Most likely the article happens to contain
  many `<Mon DD> • <author>` substrings (e.g., a meta-post listing
  other people's posts). Use `--no-drop-listing-pages` and add a
  manual `--drop-pattern` for the actual listing pages.
- **`js_required_stub` drops on real articles** → unlikely but possible
  if the article mentions one of the trigger phrases (`"This site
  requires JavaScript"`) literally. The size gate (`max_chars=600`)
  usually prevents this; if not, use `--no-drop-js-stubs`.

## Output format

For input `webscraper/corpus1/foo.txt`, output is
`webscraper/corpus1_cleaned/foo_cleaned.txt`. The `_cleaned` suffix
makes it safe to point the cleaner at its own output dir — already-
cleaned files are skipped.

## Where to go next

Once you have a directory of `_cleaned.txt` files, proceed to
[03-ingest.md](03-ingest.md).
