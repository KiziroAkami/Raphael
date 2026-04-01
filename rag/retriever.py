import logging

from sentence_transformers import SentenceTransformer
from rag.indexer import get_collection, EMBED_MODEL

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
GENERIC_PAGE_BLOCKLIST: frozenset[str] = frozenset({
    "Tensura: Reincarnated Wiki/about",
    "Tensura: Reincarnated Wiki",
    "Abilities",
    "Mobs",
    "Config",
    "Commands",
    "Crafting",
    "Skills",
    "Magic",
    "Races",
    "Items",
})

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
    Returns an empty list if no matching page is found.
    """
    candidates = list(dict.fromkeys([entity, entity.title()]))  # deduplicate, preserve order
    for candidate in candidates:
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

    # Anchor the query to the mod domain to avoid drifting toward anime content
    anchored = QUERY_PREFIX + question
    embedding = embedder.encode(anchored).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]  # cosine distance: 0 = identical, 2 = opposite

    for text, meta, dist in zip(documents, metadatas, distances):
        # Convert cosine distance to similarity score (1 = perfect, 0 = unrelated)
        score = 1.0 - (dist / 2.0)
        if score < RELEVANCE_THRESHOLD:
            continue
        page_title = meta.get("page_title", "")
        if page_title in GENERIC_PAGE_BLOCKLIST:
            logger.debug("Filtered generic page: %r (score=%.3f)", page_title, score)
            continue
        chunks.append({
            "text": text,
            "page_title": page_title,
            "section": meta.get("section", ""),
            "url": meta.get("url", ""),
            "score": round(score, 3),
        })

    return chunks
