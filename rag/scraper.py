import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import mwclient
import requests

WIKI_HOST = "tensura.wiki.gg"
WIKI_PATH = "/"
_DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_DIR = _DATA_DIR / "pages"
TIMESTAMP_FILE = _DATA_DIR / "last_updated.txt"
logger = logging.getLogger(__name__)
RATE_DELAY = 1.5        # seconds between successful requests
RETRY_DELAYS = [30, 60, 120]  # seconds to wait on successive 429s


def connect() -> mwclient.Site:
    return mwclient.Site(WIKI_HOST, path=WIKI_PATH)


def _cache_path(title: str) -> Path:
    safe = title.replace("/", "_").replace(" ", "_")
    resolved = (CACHE_DIR / f"{safe}.json").resolve()
    if resolved.parent != CACHE_DIR.resolve():
        raise ValueError(f"Path traversal detected in page title: {title!r}")
    return resolved


def _fetch_wikitext(site: mwclient.Site, title: str) -> str:
    """Fetch a single page's wikitext, retrying with backoff on 429."""
    for attempt, wait in enumerate(RETRY_DELAYS, 1):
        try:
            return site.pages[title].text()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                logger.warning(
                    "429 rate limit (HTTP) — waiting %ds before retry %d/%d...",
                    wait, attempt, len(RETRY_DELAYS),
                )
                time.sleep(wait)
            else:
                raise
        except mwclient.errors.APIError as e:
            if e.code == "ratelimited":
                logger.warning(
                    "429 rate limit (API) — waiting %ds before retry %d/%d...",
                    wait, attempt, len(RETRY_DELAYS),
                )
                time.sleep(wait)
            else:
                raise
    # Final attempt after all backoffs
    return site.pages[title].text()


def fetch_all_pages(force_refresh: bool = False) -> list[dict]:
    """Fetch every page from the wiki, using disk cache where available.

    Returns a list of dicts with keys: title, wikitext, url.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    site = connect()

    all_titles = [page.name for page in site.allpages()]
    logger.info("Found %d pages", len(all_titles))

    results = []
    for i, title in enumerate(all_titles, 1):
        path = _cache_path(title)
        if path.exists() and not force_refresh:
            data = json.loads(path.read_text(encoding="utf-8"))
            results.append(data)
            continue

        try:
            wikitext = _fetch_wikitext(site, title)
            data = {
                "title": title,
                "wikitext": wikitext,
                "url": f"https://{WIKI_HOST}/wiki/{title.replace(' ', '_')}",
            }
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            results.append(data)
            logger.info("[%d/%d] Fetched: %s", i, len(all_titles), title)
            time.sleep(RATE_DELAY)
        except Exception as e:
            logger.error("[%d/%d] ERROR fetching %r: %s", i, len(all_titles), title, e)

    logger.info("Scrape complete. %d pages ready.", len(results))
    return results


def get_last_updated() -> str | None:
    """Return the ISO timestamp of the last successful scrape, or None."""
    if TIMESTAMP_FILE.exists():
        return TIMESTAMP_FILE.read_text(encoding="utf-8").strip()
    return None


def set_last_updated(ts: str | None = None) -> None:
    """Write the current UTC time (or a given timestamp) to the timestamp file."""
    if ts is None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    TIMESTAMP_FILE.parent.mkdir(parents=True, exist_ok=True)
    TIMESTAMP_FILE.write_text(ts, encoding="utf-8")


def fetch_updated_pages(since: str) -> list[dict]:
    """Fetch only pages that changed on the wiki since the given ISO timestamp.

    Uses the MediaWiki recentchanges API — much faster than a full re-scrape.
    Returns a list of page dicts (same format as fetch_all_pages).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    site = connect()

    logger.info("Querying recentchanges since %s", since)
    # Query each type separately and merge — passing multiple types as a list
    # to mwclient reduces results instead of expanding them (API/library bug).
    changed_titles: set[str] = set()
    for rc_type in ("edit", "new", "log"):
        for change in site.recentchanges(end=since, dir="older", prop=["title"], type=[rc_type]):
            changed_titles.add(change["title"])

    logger.info("recentchanges returned %d title(s): %s", len(changed_titles), ", ".join(sorted(changed_titles)) or "(none)")
    if not changed_titles:
        logger.info("No pages changed since %s.", since)
        return []

    logger.info("Found %d changed page(s): %s", len(changed_titles), ", ".join(sorted(changed_titles)))

    results = []
    for i, title in enumerate(sorted(changed_titles), 1):
        try:
            wikitext = _fetch_wikitext(site, title)
            data = {
                "title": title,
                "wikitext": wikitext,
                "url": f"https://{WIKI_HOST}/wiki/{title.replace(' ', '_')}",
            }
            path = _cache_path(title)
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            results.append(data)
            logger.info("[%d/%d] Re-fetched: %s", i, len(changed_titles), title)
            time.sleep(RATE_DELAY)
        except Exception as e:
            logger.error("[%d/%d] ERROR fetching %r: %s", i, len(changed_titles), title, e)

    return results


def fetch_missing_pages() -> list[dict]:
    """Fetch pages that exist on the wiki but are absent from the local disk cache.

    Compares current wiki page titles against cached filenames to catch pages
    that predate the stored timestamp and are therefore invisible to recentchanges.
    Returns a list of page dicts (same format as fetch_all_pages).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    site = connect()

    wiki_titles = {page.name for page in site.allpages()}
    missing_titles = {title for title in wiki_titles if not _cache_path(title).exists()}

    logger.info(
        "Page-list diff: wiki=%d, cached=%d, missing=%d",
        len(wiki_titles), len(wiki_titles) - len(missing_titles), len(missing_titles),
    )
    if not missing_titles:
        return []

    logger.info("Missing page(s) to fetch: %s", ", ".join(sorted(missing_titles)))

    results = []
    for i, title in enumerate(sorted(missing_titles), 1):
        try:
            wikitext = _fetch_wikitext(site, title)
            data = {
                "title": title,
                "wikitext": wikitext,
                "url": f"https://{WIKI_HOST}/wiki/{title.replace(' ', '_')}",
            }
            path = _cache_path(title)
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            results.append(data)
            logger.info("[%d/%d] Fetched missing: %s", i, len(missing_titles), title)
            time.sleep(RATE_DELAY)
        except Exception as e:
            logger.error("[%d/%d] ERROR fetching missing %r: %s", i, len(missing_titles), title, e)

    return results


def prune_stale_redirects() -> list[dict]:
    """Detect cached pages with full content that are now redirects on the wiki.

    Compares local cache against the wiki's redirect list to find stale files
    left behind by page renames.  Re-fetches those pages so the cache reflects
    the current redirect state.  Returns the updated page dicts (callers should
    pass them through reindex_page to clean up ChromaDB).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    site = connect()

    wiki_redirects = {page.name for page in site.allpages(filterredir="redirects")}
    logger.info("Wiki reports %d redirect page(s)", len(wiki_redirects))

    stale: list[str] = []
    for path in CACHE_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Could not parse cache file %s: %s", path, e)
            continue
        title = data.get("title", "")
        wikitext = data.get("wikitext", "")
        if title in wiki_redirects and not wikitext.strip().upper().startswith("#REDIRECT"):
            stale.append(title)

    if not stale:
        logger.info("No stale redirect cache files found.")
        return []

    logger.info(
        "Found %d stale cache file(s) (now redirects): %s",
        len(stale), ", ".join(sorted(stale)),
    )

    results: list[dict] = []
    for i, title in enumerate(sorted(stale), 1):
        try:
            wikitext = _fetch_wikitext(site, title)
            data = {
                "title": title,
                "wikitext": wikitext,
                "url": f"https://{WIKI_HOST}/wiki/{title.replace(' ', '_')}",
            }
            _cache_path(title).write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8",
            )
            results.append(data)
            logger.info("[%d/%d] Updated stale redirect: %s", i, len(stale), title)
            time.sleep(RATE_DELAY)
        except Exception as e:
            logger.error("[%d/%d] ERROR updating %r: %s", i, len(stale), title, e)

    return results


def load_cached_pages() -> list[dict]:
    """Load all previously cached pages from disk without hitting the API."""
    if not CACHE_DIR.exists():
        return []
    pages = []
    for path in CACHE_DIR.glob("*.json"):
        try:
            pages.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            logger.warning("Could not read %s: %s", path, e)
    return pages
