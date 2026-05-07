"""Allow ``python -m seedsoftruth.webscraper`` to invoke the CLI."""

import sys

from .web_scraper import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
