# Seeds of Truth web scraper

A multi-threaded web crawler for building a Seeds of Truth corpus. Given a
seed URL and a list of "home" domains you want to fully crawl, it descends
through HTML pages and PDFs, extracts clean text, de-duplicates by content
hash, persists progress to a SQLite cache, and resumes if interrupted.

This is the rewritten descendant of the project's earlier `web_scraper_mt.py`
crawler. It preserves the original's behavior — depth model, content
de-duplication, Wayback Machine fallback, on-disk corpus layout, SQLite
cache schema — while restructuring the code so it can be used as a
self-contained module by anyone.

## Install

```
pip install -r seedsoftruth/webscraper/requirements.txt
```

For JavaScript-rendered pages (the default crawl strategy), also install
Playwright and its Chromium browser:

```
pip install -r seedsoftruth/webscraper/requirements-playwright.txt
playwright install chromium
```

If Playwright isn't installed, pass `--fetch-strategy requests` and the
crawler will use plain HTTP only.

## Usage

### CLI

```
# Generic crawl
python -m seedsoftruth.webscraper \
    --start-url https://example.com \
    --output-dir ./corpus \
    --max-pages 200

# Reproduce the original PEERS-family crawl
python -m seedsoftruth.webscraper \
    --profile peers \
    --start-url http://www.momentoflove.org \
    --output-dir ./corpus \
    --max-pages 5000

# Quick smoke test on a single page (don't follow any links)
python -m seedsoftruth.webscraper \
    --start-url https://example.com \
    --output-dir ./corpus \
    --only-root --max-pages 1
```

Run `python -m seedsoftruth.webscraper --help` for the full flag list.

### Python API

```python
from seedsoftruth.webscraper import ScraperConfig, crawl_site

cfg = ScraperConfig(
    start_url="https://example.com",
    output_dir="./corpus",
    home_domains=["example.com"],
    max_pages=200,
    workers=8,
    fetch_strategy="requests",  # or "playwright"
)
result = crawl_site(cfg)
print(f"visited {result.num_pages_visited} pages")
```

You can also load a YAML profile and override fields at construction time:

```python
cfg = ScraperConfig.from_yaml(
    "seedsoftruth/webscraper/profiles/peers.yaml",
    start_url="http://www.momentoflove.org",
    max_pages=1000,
)
```

## Profiles

Profiles are YAML files describing a crawl scope: `home_domains`,
`deny_substrings`, `deny_patterns`, plus a default `max_depth`. They're
loaded with `--profile <name>` (resolved against `webscraper/profiles/`)
or `--profile path/to/file.yaml`.

The shipped `peers` profile reproduces the editorial allowlist /
denylist that the original crawler hard-coded — it's there as both a
working example and a way to reproduce the original PEERS corpus.

## How the depth model works

- A URL whose host is in `home_domains` is a "home" URL.
- When the crawler visits a home URL, its `depth_effective` resets to 0,
  and its child links are added to the queue with `depth_effective=1`,
  on up to `max_depth`.
- A URL outside `home_domains` is fetched once (so external articles
  linked from the corpus get saved) but its child links are *not*
  enqueued. This prevents the crawler from drifting off into the open web.
- `--only-root` further restricts the crawl to URLs starting with the
  exact `start_url`. Useful for one-off page captures.

## Output layout

Inside `output_dir`, every fetched URL produces files named after a
filesystem-safe form of the URL:

| File         | Purpose                                                    |
| ------------ | ---------------------------------------------------------- |
| `<key>.html` | Raw HTML (prettified) for the page.                        |
| `<key>.txt`  | Extracted article text (newspaper3k, with soup fallback).  |
| `<key>.pdf`  | Original PDF, if the URL pointed to a PDF.                 |
| `archived_*` | Wayback fallback when the live URL returned a 4xx/5xx.     |

## Cache schema

`cache_db_path` (default `./db_cache/meta_cache.db`) is a SQLite database
with two tables:

- `downloads` — one row per URL: `cleaned_url`, `content_type`,
  `url_file_path`, `url_file_size`, `text_file_path`, `text_file_size`,
  `hash`, `download_time`. Used to skip re-downloading content already
  on disk and to detect when the on-disk file no longer matches.
- `url_queue` — pending URLs from the most recent run, so a crash or
  Ctrl-C can be resumed by simply re-running the crawler.

The DB is opened in WAL mode with a 5s busy timeout, which is the
recommended setup for a multi-threaded SQLite writer.

## Differences from the original `web_scraper_mt.py`

If you're coming from the original code:

- `config.py` constants are gone. Everything is on `ScraperConfig`.
- The hard-coded "peers family" allowlist now lives in
  `profiles/peers.yaml`. Default home-domain set is `[host_of_start_url]`.
- `is_peers_family` is now `is_home_domain`.
- Logging goes through `logging.getLogger(__name__)`. The crawler does
  not redirect `sys.stdout` anymore.
- Several races have been closed: `visited` and the bloom filter are now
  guarded by their respective locks; per-URL exceptions are logged but
  no longer kill a worker; only the "max pages hit" condition signals a
  crawl-wide shutdown.
- The unused `web_scraper_old` import is gone, along with the duplicate
  `get_txt_file_name` definition in the old `utils.py`.
- Playwright is optional. If `--fetch-strategy playwright` is selected
  but the package isn't installed, you get a clear error pointing at the
  optional requirements file.

The on-disk corpus layout and the cache DB schema are unchanged, so an
existing corpus and cache from the old crawler can be picked up by this
one.
