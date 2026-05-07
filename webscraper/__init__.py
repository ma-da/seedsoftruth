"""
Seeds of Truth web scraper.

Public API:

    >>> from seedsoftruth.webscraper import ScraperConfig, crawl_site
    >>> cfg = ScraperConfig(start_url="https://example.com",
    ...                     output_dir="./corpus",
    ...                     max_pages=50)
    >>> result = crawl_site(cfg)
    >>> result.num_pages_visited
    50

Or from the command line:

    python -m seedsoftruth.webscraper --start-url https://example.com \\
        --output-dir ./corpus --max-pages 50

See README.md in this directory for the full CLI surface and the
``--profile peers`` reference profile.
"""

from .web_scraper import CrawlResult, ScraperConfig, crawl_site, main

__all__ = ["ScraperConfig", "CrawlResult", "crawl_site", "main"]
