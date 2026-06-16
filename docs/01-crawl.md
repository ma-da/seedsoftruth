# Stage 1 — Crawling the Web

This stage takes a starting URL or sitemap and produces a directory of
plain-text files, one per source article. Everything in this stage lives
under `webscraper/`.

## What you'll need to know

A **web crawler** (or *scraper*, or *spider*) is a program that fetches
HTML pages, follows links inside them, and saves the results. Two things
make this harder than it sounds:

1. **JavaScript.** Most modern sites are *client-rendered*: the HTML the
   server returns is a thin shell, and the actual content gets loaded by
   JavaScript running in your browser. A naive `requests.get(url)` will
   return that shell with no body content. We solve this by driving a
   real headless browser (Playwright + Chromium).
2. **Infinite scroll.** Many sites (Substack archives, Twitter feeds,
   Medium tag pages) only render their first ~30 results in the initial
   HTML. Older items are loaded via XHR calls as the user scrolls. Even
   Playwright won't see those unless we explicitly scroll the page.

## The tools

### `webscraper/web_scraper.py`

A multi-threaded crawler. The two main switches you'll pick between are
`--fetch-strategy requests` (fast, no JS — fine for static sites) and
`--fetch-strategy playwright` (slower, renders JS — required for
Substack, Medium, most modern sites). Playwright is the default.

Key flags:

```
--start-url URL         Where to begin crawling.
--output-dir DIR        Where to drop the .html and .txt files.
--max-depth N           How many link hops away from start-url to follow
                        (default 2).
--max-pages N           Stop after N pages total (default -1, unlimited).
--workers N             Concurrent fetcher threads (default 12).
--only-root             Only follow links that start with --start-url
                        (don't wander to other domains).
--scroll                Scroll each rendered HTML page to the bottom
                        repeatedly until content stops loading. NEEDED
                        for infinite-scroll archive pages.
--scroll-max-hops N     Cap scroll iterations per page (default 40).
--scroll-quiet-ms N     Wait N ms after each scroll for new content
                        (default 1500).
--clear-cache           Wipe the SQLite cache before starting (use this
                        when you've changed something upstream and want
                        a fresh fetch).
--cache-db PATH         Path to the SQLite metadata cache
                        (default: ./db_cache/meta_cache.db).
--log-file PATH         Also write logs to PATH (in addition to stderr).
```

What the crawler does for each page:

1. HEAD request to determine the content type (HTML vs PDF vs other).
2. HTML → Playwright renders the page, optionally scrolls, returns the
   final DOM. PDF → `requests.get` downloads it, then `_scraper_pdf.py`
   extracts the text. Anything else is skipped.
3. The HTML is parsed by BeautifulSoup; substack CDN scripts get
   stripped; the result is saved as `<filename>.html`.
4. `newspaper3k` (a Python article-extraction library) is run to pull
   the article body out of the page chrome; if `newspaper3k` produces
   nothing (e.g., for non-article pages) the cleaner falls back to a
   generic tag-stripping pass. The result is saved as `<filename>.txt`.
5. The page's links are queued for crawling (subject to `--only-root`,
   `--max-depth`, and a Bloom filter that suppresses duplicate-content
   pages).

Two helper tables live in `db_cache/meta_cache.db`:

- `downloads`: `(cleaned_url, content_type, url_file_path, ..., hash)`.
  Lets re-runs skip pages that haven't changed.
- `url_queue`: pending URLs at process exit, so you can resume an
  interrupted crawl.

### `webscraper/substack_scrape.sh`

A wrapper specifically for Substack publications. Substack has two
quirks that bite the generic crawler:

1. **Infinite scroll on `/archive`.** The first ~30 posts render in
   HTML; older ones load via XHR. Even with `--scroll`, you can
   sometimes miss posts at the bottom of long archives.
2. **No nav links to older posts.** Once you've exhausted the initial
   render, there's no `?page=2` URL to follow.

`substack_scrape.sh` sidesteps both by calling Substack's public JSON
archive endpoint (`/api/v1/archive?sort=new&offset=N&limit=50`) to
enumerate every post slug, then feeding those URLs one at a time into
`web_scraper.py`.

Usage:

```bash
# Scrape every post on peerservice.substack.com into ./corpus_peers/
./substack_scrape.sh -o ./corpus_peers peerservice.substack.com

# Just list URLs (e.g., for diffing against an existing corpus dir)
./substack_scrape.sh --dry-run peerservice.substack.com

# Verbose; also pass extra flags through to web_scraper.py after `--`
./substack_scrape.sh -v -o ./corpus peerservice.substack.com -- --log-level DEBUG
```

The script is idempotent — `web_scraper.py`'s SHA1 dedupe means a
re-run only re-fetches URLs whose content has actually changed.

### When to use which

| Site type | Strategy |
|---|---|
| Substack publication | `substack_scrape.sh <host>` |
| Static blog with sitemap.xml | `web_scraper.py -s <sitemap>` |
| Modern JS-heavy site | `web_scraper.py -s <url> --scroll` |
| Plain static HTML | `web_scraper.py -s <url> --fetch-strategy requests` |
| Single page | `web_scraper.py -s <url> --only-root --max-depth 0` |

If you're not sure which, start with the Playwright defaults — they're
slower but rarely wrong.

## Output format

For each crawled URL like `https://peerservice.substack.com/p/cultivating-seeds-of-truth`,
the crawler produces two files in `--output-dir`:

```
peerservice.substack.com_p_cultivating-seeds-of-truth.html
peerservice.substack.com_p_cultivating-seeds-of-truth.txt
```

The `.txt` file is what stage 2 consumes. The `.html` is kept around
for debugging — when something looks weird in the cleaned text, you can
diff the raw HTML.

The naming convention is `<cleaned_url>.<ext>` where `cleaned_url` is
the URL with the scheme stripped and `/` replaced by `_`. This makes the
filename a stable identifier you can map back to the original URL.

## Common gotchas

**Off-site links leak in.** `web_scraper.py`'s `--only-root` flag
filters child URLs by prefix-match against `--start-url`. If you pass
`peerservice.substack.com` (no scheme), the crawler will normalize that
to `http://peerservice.substack.com` — but real Substack hrefs are
`https://...`, so `startswith()` returns False and the filter behaves
unexpectedly. Pass the full `https://` URL to avoid this.

**Old pending-queue contamination.** The crawler persists unfinished
URLs to `cache.db`. If you `Ctrl-C` a run and start a new one with a
different `--start-url`, the old run's pending URLs will be replayed.
Use `--no-pending-queue` or `--clear-cache` for a clean slate.

**Substack hides older posts.** If you crawl with `web_scraper.py` and
the rendered HTML, the infinite scroll caps you at ~30 posts.
`substack_scrape.sh` uses the API and gets all of them. We learned this
the hard way — see [Group A in the post-mortem](../webscraper/README.md#known-gotchas)
if it's documented there, or just remember to use the wrapper for
Substack.

**PDFs.** PDF support is enabled by default (`enable_process_pdfs:
True`). The PDF→text quality depends heavily on the PDF — clean
"born-digital" PDFs work great, scanned PDFs may produce garbage. The
file-level filter in stage 2 catches the bad ones.

## Where to go next

Once you have a directory of `.txt` files, proceed to
[02-cleaning.md](02-cleaning.md). To see this stage running on a real
small example, jump to the [codelab](codelab.md).
