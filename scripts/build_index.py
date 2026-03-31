"""
Scrape the Tensura wiki and build/update the ChromaDB index.

Usage:
    python scripts/build_index.py               # full scrape + index (first run)
    python scripts/build_index.py --incremental # re-fetch only changed pages
    python scripts/build_index.py --cached      # re-index from disk cache, no API calls
    python scripts/build_index.py --refresh     # force re-fetch all pages then index
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.scraper import fetch_all_pages, fetch_updated_pages, get_last_updated, load_cached_pages, set_last_updated
from rag.indexer import build_index, reindex_page


def main() -> None:
    args = sys.argv[1:]

    if "--incremental" in args:
        since = get_last_updated()
        if not since:
            print("No previous run found. Run without --incremental for the first full build.")
            sys.exit(1)
        print(f"Incremental update — checking for changes since {since}...")
        updated_pages = fetch_updated_pages(since)
        if not updated_pages:
            print("Index is already up to date.")
            set_last_updated()
            return
        print(f"\nRe-indexing {len(updated_pages)} changed page(s)...")
        for page in updated_pages:
            print(f"  Updating: {page['title']}")
            reindex_page(page)
        set_last_updated()
        print(f"Incremental update complete. {len(updated_pages)} page(s) refreshed.")

    elif "--cached" in args:
        print("Loading pages from disk cache...")
        pages = load_cached_pages()
        if not pages:
            print("No cached pages found. Run without --cached to fetch from wiki.")
            sys.exit(1)
        print(f"\nBuilding index for {len(pages)} pages...")
        build_index(pages)
        set_last_updated()

    else:
        force_refresh = "--refresh" in args
        print("Fetching pages from wiki..." + (" (force refresh)" if force_refresh else ""))
        pages = fetch_all_pages(force_refresh=force_refresh)
        if not pages:
            print("No pages to index.")
            sys.exit(1)
        print(f"\nBuilding index for {len(pages)} pages...")
        build_index(pages)
        set_last_updated()

    print("\nDone. Verify with:")
    print("  python -c \"from rag.indexer import get_collection; c = get_collection(); print(len(c.get()['ids']), 'chunks')\"")


if __name__ == "__main__":
    main()
