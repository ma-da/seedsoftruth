"""
Fetch backends for the web scraper.

Two strategies are exposed via a small protocol-style interface:

    RequestsFetcher    plain `requests.get`. Always available.
    PlaywrightFetcher  HEAD-then-GET (PDFs) or full Chromium render (HTML).
                       Lazily imports playwright; raises a clear error
                       if the optional dependency is not installed.

Both return a uniform tuple:

    (cleaned_url, status_code, content_type, content_bytes, was_cached)

where `was_cached` is always False from a fetcher (the cache layer sets
True before the fetcher is consulted).

Also exports:
    get_normal_traffic_headers(user_agents) -> dict
    get_wayback_url(url) -> Optional[str]
"""

from __future__ import annotations

import logging
import random
from typing import List, Optional, Tuple

import requests

from ._scraper_extract import clean_url

log = logging.getLogger(__name__)

FetchResult = Tuple[str, int, str, bytes, bool]

DEFAULT_USER_AGENTS: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
]

# Wayback Machine availability endpoint
WAYBACK_API = "http://archive.org/wayback/available"


def get_normal_traffic_headers(user_agents: Optional[List[str]] = None) -> dict:
    """Return a request header set that mimics ordinary browser traffic."""
    pool = user_agents or DEFAULT_USER_AGENTS
    return {
        "User-Agent": random.choice(pool),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
                  "image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }


def get_wayback_url(url: str, timeout: int = 10) -> Optional[str]:
    """Look up the closest archived Wayback snapshot of `url`, or None."""
    try:
        resp = requests.get(WAYBACK_API, params={"url": url}, timeout=timeout)
    except Exception as e:
        log.debug("wayback lookup error for %s: %s", url, e)
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    snap = data.get("archived_snapshots", {}).get("closest")
    if snap and snap.get("available"):
        return snap.get("url")
    return None


# --------------------------------------------------------------------------- #
# fetchers                                                                    #
# --------------------------------------------------------------------------- #

class RequestsFetcher:
    """Simple `requests.get` fetcher. Always available."""

    name = "requests"

    def fetch(self, url: str, headers: Optional[dict] = None,
              timeout: int = 15) -> FetchResult:
        cleaned = clean_url(url)
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
        except requests.exceptions.Timeout:
            log.error("timeout fetching %s", url)
            return cleaned, 408, "", b"", False
        except Exception as e:
            log.error("error fetching %s: %s", url, e)
            return cleaned, 500, "", b"", False

        return (
            cleaned,
            resp.status_code,
            resp.headers.get("Content-Type", ""),
            resp.content,
            False,
        )


class PlaywrightFetcher:
    """Playwright-based fetcher.

    Uses an HTTP HEAD probe to dispatch: PDFs are downloaded directly with
    `requests` (Chromium can't easily return PDF bytes); HTML is rendered
    in a headless Chromium so JS-rendered pages produce useful content.
    Other content types are rejected with status 415.

    The `playwright` package is imported lazily so this module can be
    imported in environments where Playwright is not installed.
    """

    name = "playwright"

    def __init__(self) -> None:
        try:
            from playwright.sync_api import sync_playwright  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "PlaywrightFetcher requires the 'playwright' package. "
                "Install with: pip install playwright && playwright install chromium"
            ) from e

    def fetch(self, url: str, headers: Optional[dict] = None,
              timeout_ms: int = 60000) -> FetchResult:
        cleaned = clean_url(url)
        # Cap requests-side timeouts to a reasonable seconds value.
        req_timeout = max(5, min(60, timeout_ms // 1000))

        # 1) HEAD probe to identify content type cheaply.
        try:
            head = requests.head(url, headers=headers,
                                 timeout=req_timeout, allow_redirects=True)
        except Exception as e:
            log.error("HEAD probe failed for %s: %s", url, e)
            return cleaned, 500, "text/html", b"", False

        content_type = (head.headers.get("Content-Type", "") or "").lower()

        if head.status_code == 429:
            return cleaned, 429, content_type, b"", False

        # 2) PDFs: direct GET via requests (Playwright can't surface bytes easily).
        if "application/pdf" in content_type:
            try:
                resp = requests.get(url, headers=headers, timeout=req_timeout)
            except Exception as e:
                log.error("PDF GET failed for %s: %s", url, e)
                return cleaned, 500, content_type, b"", False
            return (cleaned, resp.status_code, content_type,
                    resp.content if resp.status_code == 200 else b"", False)

        # 3) HTML: render in a headless browser.
        if "text/html" in content_type:
            return self._fetch_html(url, cleaned, timeout_ms)

        # 4) Plain text often signals soft rate-limit / interstitial; treat as 429.
        if "text/plain" in content_type:
            return cleaned, 429, content_type, b"", False

        log.debug("playwright: unsupported content-type %s for %s",
                  content_type, url)
        return cleaned, 415, content_type, b"", False

    def _fetch_html(self, url: str, cleaned: str, timeout_ms: int) -> FetchResult:
        from playwright.sync_api import sync_playwright

        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                try:
                    page = browser.new_page()
                    response = page.goto(url, timeout=timeout_ms)
                    if response is None:
                        return cleaned, 500, "text/html", b"", False
                    if response.status == 429:
                        return cleaned, 429, "text/html", b"", False
                    if response.status != 200:
                        return cleaned, response.status, "text/html", b"", False

                    ctype = response.headers.get("content-type", "") or "text/html"
                    if "text/html" not in ctype:
                        return cleaned, 415, ctype, b"", False
                    html = page.content()
                    return cleaned, 200, ctype, html.encode("utf-8"), False
                finally:
                    browser.close()
        except Exception as e:
            log.error("playwright render failed for %s: %s", url, e)
            return cleaned, 500, "text/html", b"", False


def make_fetcher(strategy: str) -> "RequestsFetcher | PlaywrightFetcher":
    """Return a fetcher for the named strategy ('requests' or 'playwright')."""
    if strategy == "requests":
        return RequestsFetcher()
    if strategy == "playwright":
        return PlaywrightFetcher()
    raise ValueError(f"Unknown fetch strategy: {strategy!r} "
                     "(expected 'requests' or 'playwright')")
