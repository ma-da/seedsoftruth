"""
Multi-threaded web scraper for the Seeds of Truth project.

Public surface
--------------

    ScraperConfig            dataclass holding every knob the crawler reads.
    crawl_site(config)       run a crawl. Returns CrawlResult(num_pages_visited).
    main(argv)               CLI entry point. ``python -m seedsoftruth.webscraper``.

The crawler descends into URLs whose host is in ``config.home_domains``
("home" sites — these are the corpus you intend to index) up to
``max_depth`` hops. URLs outside that set are still fetched once (so
externally-linked articles get saved) but their child links are not
traversed.

This module is the rewritten descendant of the project's original
``web_scraper_mt.py`` and the ``cache.py`` / ``utils.py`` /
``content_filter.py`` / ``pdf_fetcher.py`` modules that supported it. The
behavior is preserved; the structural changes are documented in the
adjacent README.md.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pybloom_live import BloomFilter

from . import _scraper_cache as cache
from . import _scraper_pdf as pdf
from ._scraper_extract import (
    body_adjustments,
    clean_url,
    extract_content_newspaper,
    hash_html_content,
    save_extracted_text_to_file,
)
from ._scraper_fetch import (
    DEFAULT_USER_AGENTS,
    get_normal_traffic_headers,
    get_wayback_url,
    make_fetcher,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# config                                                                      #
# --------------------------------------------------------------------------- #

# URL fragments matching truly generic skips (mailto:, javascript:, images).
# Profile-specific editorial skips are layered on top via ScraperConfig.
_HASH_TAIL_RE = re.compile(r"#([\w-]+)$")
_GENERIC_SKIP_RE = re.compile(
    r"^(?:javascripts?:|mailto:|tel:\+)|"
    r"\.(?:jpe?g|gif|png|webp|svg|ico)(?:\?|$)",
    re.IGNORECASE,
)


@dataclass
class ScraperConfig:
    """Every knob the crawler reads.

    The only required field is ``start_url``. Everything else has a
    sensible default. For the original PEERS crawl, load
    ``profiles/peers.yaml`` via ``ScraperConfig.from_yaml()``.
    """

    start_url: str
    output_dir: str = "./corpus"
    cache_db_path: str = "./db_cache/meta_cache.db"
    log_path: Optional[str] = None  # None = stderr only

    # Crawl scope.
    home_domains: List[str] = field(default_factory=list)
    deny_substrings: List[str] = field(default_factory=list)
    deny_patterns: List[str] = field(default_factory=list)
    max_depth: int = 2
    max_pages: int = -1
    only_root: bool = False

    # Concurrency / behavior.
    workers: int = 12
    fetch_strategy: str = "playwright"   # "requests" or "playwright"
    fetch_timeout_s: int = 15
    fetch_timeout_ms: int = 60000
    progress_report_n_pages: int = 25

    # Rate-limit handling (HTTP 429).
    ratelimit_retries: int = 10
    ratelimit_retry_secs: int = 10
    ratelimit_retry_incr_secs: int = 15

    # Output toggles.
    save_html_content: bool = True
    cache_enabled: bool = True
    flush_cache_on_start: bool = False
    load_pending_queue_on_start: bool = True
    enable_process_pdfs: bool = True

    # User-agent rotation pool. Empty -> built-in default pool.
    user_agents: List[str] = field(default_factory=list)

    # ---- factory helpers -------------------------------------------------- #

    @classmethod
    def from_yaml(cls, path: str | os.PathLike, **overrides) -> "ScraperConfig":
        """Build a config from a YAML profile, optionally with overrides.

        ``start_url`` may be supplied either in the YAML or via overrides.
        Unknown keys in the YAML are ignored with a warning.
        """
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "ScraperConfig.from_yaml requires PyYAML. "
                "Install with: pip install PyYAML"
            ) from e

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        data.update(overrides)

        valid = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data) - valid
        for k in unknown:
            log.warning("ignoring unknown config key %r in %s", k, path)
        clean = {k: v for k, v in data.items() if k in valid}
        return cls(**clean)


@dataclass
class CrawlResult:
    num_pages_visited: int


# --------------------------------------------------------------------------- #
# URL filtering                                                               #
# --------------------------------------------------------------------------- #

def _is_home_domain(url: str, home_domains: Set[str]) -> bool:
    """True if the host of `url` (sans leading 'www.') is in `home_domains`."""
    if not home_domains:
        return False
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host in home_domains


def _should_visit(url: str, visited: Set[str], deny_substrings: List[str],
                  deny_patterns: List[re.Pattern]) -> bool:
    if _GENERIC_SKIP_RE.search(url):
        return False
    if url in visited:
        return False

    parsed = urlparse(url)
    if parsed.netloc.lower().startswith("web.archive.org"):
        return False

    for s in deny_substrings:
        if s in url:
            return False
    for pat in deny_patterns:
        if pat.search(url):
            return False
    return True


def _should_process_child_links(child_depth: int, is_home: bool,
                                max_depth: int) -> bool:
    """We only descend into child links inside home domains, up to max_depth."""
    if child_depth >= max_depth:
        return False
    return is_home


# --------------------------------------------------------------------------- #
# crawl                                                                       #
# --------------------------------------------------------------------------- #

def crawl_site(config: ScraperConfig) -> CrawlResult:
    """Run a multi-threaded crawl. Returns a CrawlResult."""
    # Resolve home domains: default to the host of start_url if none given.
    home_set: Set[str] = set(d.lower().lstrip(".") for d in config.home_domains)
    if not home_set:
        host = urlparse(config.start_url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        home_set = {host} if host else set()
        log.info("home_domains not configured; defaulting to {%s}", host)

    deny_patterns: List[re.Pattern] = [re.compile(p) for p in config.deny_patterns]
    user_agents = config.user_agents or DEFAULT_USER_AGENTS

    # Working dirs and DB.
    os.makedirs(config.output_dir, exist_ok=True)
    cache.init_db(config.cache_db_path)
    if not cache.table_exists(config.cache_db_path, "downloads"):
        raise RuntimeError("cache db missing 'downloads' table after init")

    if config.flush_cache_on_start:
        cache.clear_downloads(config.cache_db_path)
        log.info("flushed cache on start")

    # Build fetcher (raises if Playwright is requested but not installed).
    fetcher = make_fetcher(config.fetch_strategy)
    log.info("using fetch strategy: %s", fetcher.name)

    # Shared state.
    visited: Set[str] = set()
    visited_lock = threading.Lock()

    seen_hashes = BloomFilter(capacity=1_000_000, error_rate=0.00001)
    seen_hashes_lock = threading.Lock()

    num_visited = 0
    num_visited_lock = threading.Lock()
    next_progress_at = config.progress_report_n_pages

    url_queue: Queue = Queue()
    stop_event = threading.Event()
    max_pages_hit = threading.Event()  # specifically signals "we're done by limit"

    # ---- helpers -------------------------------------------------------- #

    def add_url_to_crawl(url: str, depth_actual: int, depth_effective: int) -> None:
        url_queue.put((url, depth_actual, depth_effective))
        cache.save_pending_url(config.cache_db_path, url,
                               depth_actual, depth_effective)

    def finalize_url(url: str) -> None:
        cache.delete_pending_url(config.cache_db_path, url)

    def fetch_with_ratelimit(url: str, headers: dict):
        """Fetch with retry on 429. Returns the fetcher tuple plus was_cached."""
        cleaned = clean_url(url)
        # cache hit?
        if config.cache_enabled:
            cached = cache.get_cached_file_content(config.cache_db_path, cleaned)
            if cached is not None:
                content, ctype = cached
                return cleaned, 200, ctype, content, True

        # uncached: fetch with rate-limit handling.
        if config.fetch_strategy == "playwright":
            result = fetcher.fetch(url, headers=headers,
                                   timeout_ms=config.fetch_timeout_ms)
        else:
            result = fetcher.fetch(url, headers=headers,
                                   timeout=config.fetch_timeout_s)

        retries = 0
        while result[1] == 429 and retries < config.ratelimit_retries:
            backoff = (config.ratelimit_retry_secs
                       + retries * config.ratelimit_retry_incr_secs)
            log.warning("429 from %s; backing off %ds (retry %d)",
                        url, backoff, retries + 1)
            time.sleep(backoff)
            headers = get_normal_traffic_headers(user_agents)
            if config.fetch_strategy == "playwright":
                result = fetcher.fetch(url, headers=headers,
                                       timeout_ms=config.fetch_timeout_ms)
            else:
                result = fetcher.fetch(url, headers=headers,
                                       timeout=config.fetch_timeout_s)
            retries += 1
        if result[1] == 429:
            log.error("exhausted rate-limit retries for %s", url)
        return result

    def crawl(url: str, depth_actual: int, depth_effective: int) -> None:
        nonlocal num_visited, next_progress_at

        is_home = _is_home_domain(url, home_set) or url.startswith(config.start_url)
        if is_home:
            depth_effective = 0  # home pages reset effective depth

        with num_visited_lock:
            num_visited += 1
            if (config.max_pages > 0) and (num_visited > config.max_pages):
                log.info("max_pages limit hit (%d); signaling shutdown",
                         config.max_pages)
                max_pages_hit.set()
                stop_event.set()
                return
            if num_visited >= next_progress_at:
                log.info("progress: visited %d pages", num_visited)
                next_progress_at += config.progress_report_n_pages

        with visited_lock:
            visited.add(url)

        log.info("(%d/%d) CRAWLING: %s", depth_actual, depth_effective, url)

        headers = get_normal_traffic_headers(user_agents)

        try:
            cleaned_url, status_code, content_type, content, was_cached = \
                fetch_with_ratelimit(url, headers)
        except requests.exceptions.Timeout:
            log.error("timeout for %s", url)
            return
        except Exception as e:
            log.error("fetch error for %s: %s", url, e)
            return

        if status_code != 200:
            _handle_broken_link(url, status_code, config, headers)
            return

        # Dispatch by content type.
        if config.enable_process_pdfs and "application/pdf" in content_type:
            _handle_pdf(url, cleaned_url, config)
            return

        if "text/html" not in content_type:
            log.debug("skipping non-HTML/PDF (%s) %s", content_type, url)
            return

        # HTML.
        try:
            soup = BeautifulSoup(content, "html.parser")
        except Exception as e:
            log.error("HTML parse failed for %s: %s", url, e)
            return
        body_adjustments(soup)
        soup_bytes = soup.prettify().encode()

        h = hash_html_content(soup_bytes)
        with seen_hashes_lock:
            if h in seen_hashes:
                log.debug("already-seen content; skipping %s", url)
                return
            seen_hashes.add(h)

        if not was_cached:
            _save_html(cleaned_url, soup_bytes, h, config)
        else:
            # Regenerate text file if it disappeared.
            txt_filename = os.path.join(
                config.output_dir, cleaned_url.replace("/", "_") + ".txt"
            )
            if not os.path.exists(txt_filename):
                log.info("regenerating missing txt: %s", txt_filename)
                save_extracted_text_to_file(txt_filename, soup_bytes)

        # Enqueue child links.
        child_depth = depth_effective + 1
        if not _should_process_child_links(child_depth, is_home, config.max_depth):
            return

        for link in soup.find_all("a", href=True):
            child_url = urljoin(url, link["href"])
            child_url = _HASH_TAIL_RE.sub("", child_url)
            if config.only_root and not child_url.startswith(config.start_url):
                continue
            with visited_lock:
                already = child_url in visited
            if already:
                continue
            if not _should_visit(child_url, visited, config.deny_substrings,
                                 deny_patterns):
                continue
            add_url_to_crawl(child_url, depth_actual + 1, child_depth)

    def worker(worker_id: int) -> None:
        log.debug("worker %d started", worker_id)
        while not stop_event.is_set():
            try:
                url, da, de = url_queue.get(timeout=1.0)
            except queue.Empty:
                # Spin until either there's work or someone signals shutdown.
                continue

            try:
                crawl(url, da, de)
            except Exception as e:
                # Log + continue: a single bad URL shouldn't kill the worker.
                log.exception("worker %d: unhandled error on %s: %s",
                              worker_id, url, e)
            finally:
                finalize_url(url)
                url_queue.task_done()
        log.debug("worker %d exiting", worker_id)

    # ---- prep -------------------------------------------------------- #

    if config.load_pending_queue_on_start:
        added = cache.load_pending_urls(config.cache_db_path, url_queue)
        if added > 0:
            cache.clear_pending_queue(config.cache_db_path)

    add_url_to_crawl(config.start_url, 0, 0)

    log.info("starting %d workers", config.workers)
    threads = [
        threading.Thread(target=worker, args=(i + 1,), daemon=True,
                         name=f"scraper-worker-{i + 1}")
        for i in range(config.workers)
    ]
    for t in threads:
        t.start()

    # Wait for either queue drain or a stop signal.
    try:
        while not stop_event.is_set():
            if url_queue.unfinished_tasks == 0:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        log.warning("interrupt received; signaling shutdown")
        stop_event.set()

    stop_event.set()
    for t in threads:
        t.join(timeout=10)

    if max_pages_hit.is_set():
        log.info("crawl ended: max_pages hit")
    log.info("** crawl finished, visited %d pages", num_visited)
    return CrawlResult(num_pages_visited=num_visited)


# --------------------------------------------------------------------------- #
# small per-content-type helpers                                              #
# --------------------------------------------------------------------------- #

def _save_html(cleaned_url: str, soup_bytes: bytes, content_hash: str,
               config: ScraperConfig) -> None:
    if not config.save_html_content:
        return
    base = os.path.join(config.output_dir, cleaned_url.replace("/", "_"))
    html_path = base + ".html"
    txt_path = base + ".txt"
    with open(html_path, "wb") as f:
        f.write(soup_bytes)
    txt_saved = save_extracted_text_to_file(txt_path, soup_bytes)
    if config.cache_enabled:
        url_size = os.path.getsize(html_path)
        txt_size = os.path.getsize(txt_path) if txt_saved else 0
        cache.update_cache(
            config.cache_db_path,
            cleaned_url=cleaned_url,
            content_type="text/html",
            url_file_path=html_path,
            url_file_size=url_size,
            text_file_path=txt_path if txt_saved else "",
            text_file_size=txt_size,
            content_hash=content_hash,
        )
    log.debug("saved HTML+TXT for %s", cleaned_url)


def _handle_pdf(url: str, cleaned_url: str, config: ScraperConfig) -> None:
    log.info("PDF detected: %s", url)
    pdf_path = os.path.join(config.output_dir,
                            cleaned_url.replace("/", "_") + ".pdf")
    txt_path = pdf_path.replace(".pdf", ".txt")

    if not pdf.download_pdf(url, pdf_path):
        return
    try:
        title, text = pdf.extract_clean_pdf_text(pdf_path)
    except ImportError as e:
        log.error("PDF extraction unavailable: %s", e)
        return
    except Exception as e:
        log.error("PDF extraction failed for %s: %s", pdf_path, e)
        return
    pdf.save_text_to_file(title, text, txt_path)


def _handle_broken_link(url: str, status_code: int, config: ScraperConfig,
                        headers: dict) -> None:
    log.warning("broken link %s (status=%s); checking Wayback", url, status_code)
    archived = get_wayback_url(url)
    if not archived:
        log.error("no archived snapshot for %s", url)
        return
    log.info("retrieving archived version: %s", archived)

    # Filename-safe key for the archived copy.
    cleaned = clean_url(archived)
    cleaned = (cleaned.replace("?", "QQ").replace("=", "EQ").replace("&", "AMP"))
    out_base = os.path.join(config.output_dir, "archived_" + cleaned.replace("/", "_"))
    try:
        resp = requests.get(archived, headers=headers,
                            timeout=config.fetch_timeout_s)
    except Exception as e:
        log.error("archive fetch failed for %s: %s", archived, e)
        return
    if resp.status_code != 200:
        log.error("archive returned %s for %s", resp.status_code, archived)
        return
    # Best-effort: write whatever bytes came back. Suffix is unknown without
    # another HEAD round-trip, so the original archived URL extension wins.
    with open(out_base, "wb") as f:
        f.write(resp.content)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="seedsoftruth-webscraper",
        description="Multi-threaded crawler for building a Seeds of Truth corpus.",
    )
    p.add_argument("--start-url", "-s", required=False, default=None,
                   help="URL to begin crawling from. Required unless given by --profile.")
    p.add_argument("--output-dir", "-o", default=None,
                   help="Directory to write crawled HTML/TXT/PDF output.")
    p.add_argument("--cache-db", default=None,
                   help="Path to the SQLite cache db.")
    p.add_argument("--profile", default=None,
                   help="Path to a YAML profile, or a built-in profile name "
                        "(e.g. 'peers').")
    p.add_argument("--max-pages", "-m", type=int, default=None,
                   help="Stop after this many pages (-1 or 0 = unlimited).")
    p.add_argument("--max-depth", "-d", type=int, default=None,
                   help="Maximum crawl depth from a home domain.")
    p.add_argument("--workers", "-w", type=int, default=None,
                   help="Number of worker threads.")
    p.add_argument("--fetch-strategy", choices=("requests", "playwright"),
                   default=None, help="Backend used to fetch HTML.")
    p.add_argument("--only-root", action="store_true",
                   help="Only fetch start-url; do not follow any links.")
    p.add_argument("--clear-cache", action="store_true",
                   help="Wipe the downloads cache table before starting.")
    p.add_argument("--no-pending-queue", action="store_true",
                   help="Do not load any persisted pending URLs from a prior run.")
    p.add_argument("--log-file", default=None,
                   help="If set, also write logs to this file.")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                   help="Log verbosity (default: INFO).")
    return p


def _resolve_profile(profile_arg: str) -> Path:
    """Resolve a profile name or path to a YAML file."""
    candidate = Path(profile_arg)
    if candidate.is_file():
        return candidate
    bundled = Path(__file__).parent / "profiles" / f"{profile_arg}.yaml"
    if bundled.is_file():
        return bundled
    raise FileNotFoundError(
        f"profile {profile_arg!r} not found "
        f"(tried {candidate} and {bundled})"
    )


def _configure_logging(level: str, log_file: Optional[str]) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Always log to stderr.
    stderr_h = logging.StreamHandler(sys.stderr)
    stderr_h.setFormatter(fmt)
    root.addHandler(stderr_h)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".",
                    exist_ok=True)
        file_h = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_h.setFormatter(fmt)
        root.addHandler(file_h)


def _normalize_url(url: str) -> str:
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "http://" + url
    return url


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.log_level, args.log_file)

    # Build base config (from profile if given, else defaults).
    overrides = {}
    if args.start_url:
        overrides["start_url"] = _normalize_url(args.start_url)
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.cache_db:
        overrides["cache_db_path"] = args.cache_db
    if args.max_pages is not None:
        overrides["max_pages"] = args.max_pages
    if args.max_depth is not None:
        overrides["max_depth"] = args.max_depth
    if args.workers is not None:
        overrides["workers"] = args.workers
    if args.fetch_strategy is not None:
        overrides["fetch_strategy"] = args.fetch_strategy
    if args.only_root:
        overrides["only_root"] = True
    if args.no_pending_queue:
        overrides["load_pending_queue_on_start"] = False
    if args.clear_cache:
        overrides["flush_cache_on_start"] = True
        overrides["load_pending_queue_on_start"] = False

    if args.profile:
        profile_path = _resolve_profile(args.profile)
        if "start_url" not in overrides:
            # from_yaml requires start_url; let it raise a clear TypeError if missing.
            cfg = ScraperConfig.from_yaml(profile_path, **overrides)
        else:
            cfg = ScraperConfig.from_yaml(profile_path, **overrides)
    else:
        if "start_url" not in overrides:
            print("--start-url is required when no --profile is given",
                  file=sys.stderr)
            return 2
        cfg = ScraperConfig(**overrides)

    log.info("crawl begin: start_url=%s output_dir=%s",
             cfg.start_url, cfg.output_dir)
    result = crawl_site(cfg)
    log.info("crawl end: %d pages visited", result.num_pages_visited)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
