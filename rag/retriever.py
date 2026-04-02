import logging

from sentence_transformers import SentenceTransformer
from rag.indexer import get_collection, EMBED_MODEL
from llm.client import expand_query

logger = logging.getLogger(__name__)

QUERY_PREFIX = "In the Tensura Minecraft mod, "
RELEVANCE_THRESHOLD = 0.30   # chunks below this score are discarded
K_FACTUAL = 8
K_COMPARATIVE = 10

# Keywords that signal the user wants a comparison or recommendation
COMPARATIVE_KEYWORDS = {
    "best", "worst", "strongest", "weakest", "compare", "vs", "versus",
    "recommend", "recommended", "worth", "better", "worse", "which",
    "top", "ranking", "rank", "optimal", "most", "least",
}

# Pages that are broad mod overviews or navigation hubs — they score high for
# almost any query and crowd out specific content pages.
# Exact-match: only the root page is blocked (e.g. "Races" but NOT "Races/Human").
# Prefix-match: root + all subpages blocked (e.g. "Tensura: Reincarnated Wiki/*").
_EXACT_BLOCKED: frozenset[str] = frozenset({
    "Abilities",
    "Effects",
    "Mobs",
    "Config",
    "Commands",
    "Crafting",
    "Skills",
    "Magic",
    "Races",
    "Items",
})

_PREFIX_BLOCKED: tuple[str, ...] = (
    "Tensura: Reincarnated Wiki",  # blocks /welcome, /links, /contribute, /about
)


def _is_blocked(page_title: str) -> bool:
    """Return True if a page title is on the blocklist."""
    if page_title in _EXACT_BLOCKED:
        return True
    return any(
        page_title == p or page_title.startswith(p + "/")
        for p in _PREFIX_BLOCKED
    )

# Words that indicate a structured question, not a bare entity lookup
_QUESTION_STARTERS = {
    "how", "what", "why", "where", "when", "is", "are", "does",
    "can", "will", "which", "who", "whose", "whom", "do", "did",
    "has", "have", "had", "was", "were", "should", "would", "could",
    "tell", "explain", "describe", "list", "show",
}

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def is_comparative(question: str) -> bool:
    """Return True if the question appears to be asking for a comparison or recommendation."""
    words = set(question.lower().split())
    return bool(words & COMPARATIVE_KEYWORDS)


def _bare_entity_name(question: str) -> str | None:
    """Return the entity name if the question is a 1–3 word bare noun lookup, else None.

    Example: "Predator?" → "Predator", "Great Sage?" → "Great Sage"
    Example: "What is Predator?" → None (starts with question word)
    """
    text = question.rstrip("?").strip()
    words = text.split()
    if not words or len(words) > 3:
        return None
    if words[0].lower() in _QUESTION_STARTERS:
        return None
    return text


def _query_by_page_title(collection, entity: str) -> list[dict]:
    """Return all chunks for a wiki page matching the entity name.

    Tries the entity as-is and then title-cased to handle capitalisation variants.
    Returns an empty list if no matching page is found or if the page is blocked.
    """
    candidates = list(dict.fromkeys([entity, entity.title()]))  # deduplicate, preserve order
    for candidate in candidates:
        if _is_blocked(candidate):
            logger.debug("Bare entity %r is on blocklist, skipping", candidate)
            continue
        result = collection.get(
            where={"page_title": {"$eq": candidate}},
            include=["documents", "metadatas"],
        )
        if result["documents"]:
            logger.debug("Bare entity hit: page_title=%r (%d chunks)", candidate, len(result["documents"]))
            return [
                {
                    "text": text,
                    "page_title": meta.get("page_title", ""),
                    "section": meta.get("section", ""),
                    "url": meta.get("url", ""),
                    "score": 1.0,  # exact metadata match — treat as maximally relevant
                }
                for text, meta in zip(result["documents"], result["metadatas"])
            ]
    return []


def _inject_page_title_variant(
    collection, question: str, variants: list[str],
) -> None:
    """If words in the question match a page title, add it as a search variant.

    Prevents common terms ("cooldown", "damage") from drowning entity names
    ("creator", "predator") in the embedding. Checks 1-, 2-, and 3-word windows
    from the question against ChromaDB page titles.
    """
    words = question.rstrip("?").strip().split()
    # Skip question-starter words for candidate generation
    content_words = [w for w in words if w.lower() not in _QUESTION_STARTERS]
    if not content_words:
        return

    # Build candidate phrases: single words, bigrams, trigrams
    candidates: list[str] = []
    for size in (1, 2, 3):
        for i in range(len(content_words) - size + 1):
            phrase = " ".join(content_words[i:i + size])
            if len(phrase) >= 3:  # skip very short candidates
                candidates.append(phrase)

    for phrase in candidates:
        if phrase in variants:
            continue
        # Try exact page title match (case-insensitive via title-case)
        for form in dict.fromkeys([phrase, phrase.title()]):
            if _is_blocked(form):
                continue
            result = collection.get(
                where={"page_title": {"$eq": form}},
                include=["metadatas"],
                limit=1,
            )
            if result["ids"]:
                logger.debug("Page title boost: %r matches page %r", phrase, form)
                variants.append(form)
                return  # one boost is enough


def query(question: str) -> list[dict]:
    """Embed the question, search ChromaDB, and return the most relevant chunks.

    For bare entity queries (e.g. "Predator?"), attempts an exact page_title lookup
    before falling back to semantic search.

    Each returned dict has keys: text, page_title, section, url, score.
    """
    collection = get_collection()

    # Fast path: bare entity lookup (e.g. "Predator?", "Great Sage?")
    entity = _bare_entity_name(question)
    if entity:
        page_chunks = _query_by_page_title(collection, entity)
        if page_chunks:
            return page_chunks
        logger.debug("Bare entity %r not found by page_title, falling back to semantic search", entity)

    embedder = _get_embedder()
    k = K_COMPARATIVE if is_comparative(question) else K_FACTUAL

    # Expand the query to cover vocabulary mismatches (e.g. "tame" → "charm/subjugate").
    # Falls back to the original query only if expansion fails.
    variants = [question] + expand_query(question)

    # TEN-111: If bare entity lookup failed, inject the entity name as a search
    # variant so semantic search can still find the right page.
    if entity and entity not in variants:
        variants.append(entity)

    # TEN-113: If any word(s) in the query match a page title, inject that title
    # as a variant to prevent common terms from drowning entity names.
    # Runs for all queries that reach semantic search — including bare entities
    # whose exact page_title lookup failed (e.g. "creator cooldown?").
    _inject_page_title_variant(collection, question, variants)

    logger.debug("Query variants (%d): %s", len(variants), variants)

    # Run each variant through ChromaDB and merge results, keeping the highest
    # score per unique chunk (deduplication by chunk text).
    seen: dict[str, dict] = {}  # text → best chunk dict so far

    for variant in variants:
        anchored = QUERY_PREFIX + variant
        embedding = embedder.encode(anchored).tolist()

        results = collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - (dist / 2.0)  # cosine distance → similarity
            if score < RELEVANCE_THRESHOLD:
                continue
            page_title = meta.get("page_title", "")
            if _is_blocked(page_title):
                continue
            # Keep the entry only if it improves on what we already have
            if text not in seen or score > seen[text]["score"]:
                seen[text] = {
                    "text": text,
                    "page_title": page_title,
                    "section": meta.get("section", ""),
                    "url": meta.get("url", ""),
                    "score": round(score, 3),
                }

    # Return top-K sorted by score descending
    return sorted(seen.values(), key=lambda c: c["score"], reverse=True)[:k]
