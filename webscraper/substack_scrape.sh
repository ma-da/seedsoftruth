#!/usr/bin/env bash
# substack_scrape.sh — pull every article from a Substack publication.
#
# Uses Substack's public archive JSON API (/api/v1/archive) to enumerate
# every post slug — bypassing the infinite-scroll archive UI that hides
# older posts from crawlers — then drives web_scraper.py to fetch each
# article one URL at a time.
#
# This is option 1 from the post-mortem on corpus1/. Use this when you
# want guaranteed completeness on a Substack site; use ``web_scraper.py
# --scroll`` directly when you also need the listing/topic pages and are
# happy to rely on infinite-scroll rendering.

set -euo pipefail

PROG="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRAPER="${SCRIPT_DIR}/web_scraper.py"

usage() {
  cat <<EOF
$PROG — scrape every article from a Substack publication via its JSON API.

USAGE
  $PROG [options] <host>

ARGUMENTS
  <host>                    Substack hostname, e.g. peerservice.substack.com
                            (scheme optional; stripped if present)

OPTIONS
  -o, --output-dir DIR      Output directory for crawled files
                            (default: ./corpus_<host>)
      --cache-db PATH       Cache DB path passed to web_scraper.py
                            (default: ./db_cache/meta_cache.db)
      --limit N             Posts per API call; max 50 (default: 50)
      --scroll              Pass --scroll to web_scraper.py (rarely needed
                            for individual article pages, but useful if an
                            article has lazy-loaded embeds)
      --scraper PATH        Path to web_scraper.py
                            (default: $SCRAPER)
      --python BIN          Python interpreter to use (default: python3)
      --dry-run             Print URLs that would be scraped, then exit
      --list-only           Alias for --dry-run
  -v, --verbose             Echo each web_scraper.py invocation
  -h, --help                Show this help and exit

EXAMPLES
  # Scrape the whole peerservice.substack.com publication into ./corpus_peers/
  $PROG -o ./corpus_peers peerservice.substack.com

  # Just list the URLs the API exposes — useful for diffing against an
  # existing corpus directory.
  $PROG --dry-run peerservice.substack.com

  # Scrape and pass extra args to web_scraper.py after a -- separator.
  $PROG -o ./corpus peerservice.substack.com -- --log-level DEBUG

NOTES
  * The /api/v1/archive endpoint returns ALL public posts including older
    ones that the rendered /archive page hides behind infinite scroll.
  * Each article is fetched with --only-root --max-depth 0, so no link
    traversal happens — exactly one HTTP fetch per article.
  * web_scraper.py's cache means re-running this script is cheap; already-
    fetched articles are skipped automatically.
EOF
}

# ---- defaults ------------------------------------------------------------- #

OUTPUT_DIR=""
CACHE_DB="./db_cache/meta_cache.db"
LIMIT=50
SCROLL_FLAG=""
PYTHON_BIN="python3"
DRY_RUN=0
VERBOSE=0
HOST=""
EXTRA_SCRAPER_ARGS=()

# ---- arg parsing ---------------------------------------------------------- #

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)        usage; exit 0 ;;
    -o|--output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
    --cache-db)       CACHE_DB="$2"; shift 2 ;;
    --limit)          LIMIT="$2"; shift 2 ;;
    --scroll)         SCROLL_FLAG="--scroll"; shift ;;
    --scraper)        SCRAPER="$2"; shift 2 ;;
    --python)         PYTHON_BIN="$2"; shift 2 ;;
    --dry-run|--list-only) DRY_RUN=1; shift ;;
    -v|--verbose)     VERBOSE=1; shift ;;
    --)               shift; EXTRA_SCRAPER_ARGS+=("$@"); break ;;
    -*)
      echo "$PROG: unknown option: $1" >&2
      echo "Try '$PROG --help' for usage." >&2
      exit 2
      ;;
    *)
      if [[ -z "$HOST" ]]; then
        HOST="$1"; shift
      else
        echo "$PROG: unexpected positional argument: $1" >&2
        exit 2
      fi
      ;;
  esac
done

if [[ -z "$HOST" ]]; then
  echo "$PROG: missing required <host> argument" >&2
  echo "Try '$PROG --help' for usage." >&2
  exit 2
fi

# Normalize host: strip scheme + trailing slash if user pasted a URL.
HOST="${HOST#http://}"
HOST="${HOST#https://}"
HOST="${HOST%/}"

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="./corpus_${HOST}"
fi

if [[ ! -f "$SCRAPER" ]] && [[ "$DRY_RUN" -eq 0 ]]; then
  echo "$PROG: web_scraper.py not found at $SCRAPER" >&2
  echo "Pass --scraper PATH or run from the webscraper directory." >&2
  exit 1
fi

# ---- enumerate URLs via Substack archive API ------------------------------ #

# Use python for JSON; jq is not always installed.
collect_urls() {
  "$PYTHON_BIN" - "$HOST" "$LIMIT" <<'PY'
import json, sys, urllib.error, urllib.parse, urllib.request

host = sys.argv[1]
limit = int(sys.argv[2])
offset = 0
seen = set()
empty_or_all_dupes = 0     # tolerate one stray quirky page before stopping
ua = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# Note on pagination: Substack's /api/v1/archive returns variable-sized
# batches — asking for limit=50 frequently yields fewer than 50 rows even
# when there are more posts behind it. So we advance offset by the actual
# returned count and only stop when two consecutive calls deliver no new
# slugs (either empty batches or all-already-seen).
while True:
    q = urllib.parse.urlencode(
        {"sort": "new", "search": "", "offset": offset, "limit": limit}
    )
    url = f"https://{host}/api/v1/archive?{q}"
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            batch = json.load(r)
    except urllib.error.HTTPError as e:
        print(f"API error {e.code} at offset={offset}: {e.reason}",
              file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"API fetch failed at offset={offset}: {e}", file=sys.stderr)
        sys.exit(1)

    new = 0
    for post in batch:
        slug = post.get("slug")
        if not slug or slug in seen:
            continue
        seen.add(slug)
        print(f"https://{host}/p/{slug}")
        new += 1

    if not batch:
        # Genuine end-of-list. Stop unconditionally.
        break
    if new == 0:
        empty_or_all_dupes += 1
        if empty_or_all_dupes >= 2:
            break
    else:
        empty_or_all_dupes = 0

    # Advance by the size of the actual response, never by `limit`.
    offset += len(batch)
PY
}

echo "$PROG: enumerating posts from https://${HOST}/api/v1/archive ..." >&2
URLS_FILE="$(mktemp -t substack-urls.XXXXXX)"
trap 'rm -f "$URLS_FILE"' EXIT

collect_urls > "$URLS_FILE"
COUNT="$(wc -l < "$URLS_FILE" | tr -d ' ')"
echo "$PROG: found $COUNT article URL(s)" >&2

if [[ "$DRY_RUN" -eq 1 ]]; then
  cat "$URLS_FILE"
  exit 0
fi

if [[ "$COUNT" -eq 0 ]]; then
  echo "$PROG: nothing to scrape (host may not be a Substack publication)" >&2
  exit 1
fi

# ---- drive web_scraper.py per URL ----------------------------------------- #

mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$CACHE_DB")"

ok=0
fail=0
while IFS= read -r url; do
  [[ -z "$url" ]] && continue
  args=(
    "$PYTHON_BIN" "$SCRAPER"
    --start-url "$url"
    --output-dir "$OUTPUT_DIR"
    --cache-db "$CACHE_DB"
    --only-root
    --max-depth 0
  )
  [[ -n "$SCROLL_FLAG" ]] && args+=("$SCROLL_FLAG")
  if [[ ${#EXTRA_SCRAPER_ARGS[@]} -gt 0 ]]; then
    args+=("${EXTRA_SCRAPER_ARGS[@]}")
  fi
  if [[ "$VERBOSE" -eq 1 ]]; then
    echo "+ ${args[*]}" >&2
  fi
  if "${args[@]}"; then
    ok=$((ok+1))
  else
    fail=$((fail+1))
    echo "$PROG: scrape failed for $url" >&2
  fi
done < "$URLS_FILE"

echo "$PROG: done — $ok succeeded, $fail failed, output in $OUTPUT_DIR" >&2
[[ "$fail" -eq 0 ]] || exit 1
