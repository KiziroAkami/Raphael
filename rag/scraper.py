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

    # recentchanges lists newest→oldest by default; rcend is the cut-off (oldest point)
    changed_titles: set[str] = set()
    for change in site.recentchanges(end=since, dir="older", prop=["title"], type=["edit", "new"]):
        changed_titles.add(change["title"])

    if not changed_titles:
        logger.info("No pages changed since last update.")
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
