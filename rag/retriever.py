import logging
import re
import threading

from sentence_transformers import SentenceTransformer
from rag.indexer import get_collection, EMBED_MODEL
from llm.client import expand_query

logger = logging.getLogger(__name__)

QUERY_PREFIX = "In the Tensura Minecraft mod, "
RELEVANCE_THRESHOLD = 0.30   # chunks below this score are discarded
K_FACTUAL = 8
K_COMPARATIVE = 10
MAX_ENUM_CHARS = 8000        # char budget for enumeration (breadth — many entries, light stats)
MAX_COMPARATIVE_CHARS = 16000  # char budget for comparisons (depth — fewer entries, full stats)

# Condensation keyword filters — which infobox lines to keep per entry
_CONDENSATION_LIGHT = ("Type:", "HP:", "SHP:", "Obtain Cost:", "title:", "MP Range:")
_CONDENSATION_VERBOSE = (
    "Type:", "HP:", "SHP:", "Obtain Cost:", "title:", "MP Range:", "AP Range:",
    "Attack DMG:", "Attack Speed:", "Intrinsics:", "Previous:", "Next:",
    "Difficulty:", "Majin:", "Spiritual:", "Divine:", "Size:",
    "Points to Master:", "Knockback Resist:", "Speed:", "Sprint Speed:",
)

# ---------------------------------------------------------------------------
# Enumeration detection — "list all X", "what X are there", etc.
# ---------------------------------------------------------------------------

# Regex patterns that signal an enumeration intent.
_ENUM_PATTERNS = [
    re.compile(r"\b(?:list|name|show)\s+(?:all|every|each)\b", re.IGNORECASE),
    re.compile(r"\b(?:all|every)\s+(?:the\s+)?(?:available\s+)?(\w[\w\s]{1,30})\b", re.IGNORECASE),
    re.compile(r"\bhow\s+many\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+[\w\s]{0,20}(?:are\s+there|exist|does\s+the\s+mod\s+have)\b", re.IGNORECASE),
]

# Maps user-facing category terms to retrieval strategies.
# "prefix:<path>" = fetch all summary chunks whose page_title starts with <path>/
# "content:<marker>" = fetch all summary chunks whose text contains <marker>
_CATEGORY_MAP: list[tuple[re.Pattern[str], str]] = [
    # Skill types (by content marker in summary chunks)
    (re.compile(r"\buniques?\b|\bunique\s+skills?\b", re.IGNORECASE), "content:Unique Skill"),
    (re.compile(r"\bextra\s+skills?\b", re.IGNORECASE), "content:Extra Skill"),
    (re.compile(r"\bcommon\s+skills?\b", re.IGNORECASE), "content:Common Skill"),
    (re.compile(r"\bintrinsic\s+skills?\b", re.IGNORECASE), "content:Intrinsic Skill"),
    (re.compile(r"\bresistance\s+skills?\b", re.IGNORECASE), "content:Resistance Skill"),
    (re.compile(r"\bbattlewills?\b", re.IGNORECASE), "prefix:Abilities/Battlewills"),
    # Engraving sub-categories (sections within the Engravings page)
    (re.compile(r"\bblessings?\b", re.IGNORECASE), "allcontent:Blessing"),
    (re.compile(r"\bcurses?\b", re.IGNORECASE), "allcontent:Curse"),
    (re.compile(r"\bengravings?\b", re.IGNORECASE), "allcontent:Engravings —"),
    # Prefix-based categories
    (re.compile(r"\braces?\b", re.IGNORECASE), "prefix:Races"),
    (re.compile(r"\bmobs?\b", re.IGNORECASE), "prefix:Mobs"),
    (re.compile(r"\bmagics?\b|\bspells?\b", re.IGNORECASE), "prefix:Abilities/Magics"),
    (re.compile(r"\beffects?\b|\bstatus\s+effects?\b", re.IGNORECASE), "prefix:Effects"),
    (re.compile(r"\bblocks?\b", re.IGNORECASE), "prefix:Blocks"),
    (re.compile(r"\bstructures?\b", re.IGNORECASE), "prefix:Structures"),
    (re.compile(r"\bschematics?\b", re.IGNORECASE), "prefix:Items/Schematics"),
    (re.compile(r"\bitems?\b", re.IGNORECASE), "prefix:Items"),
    # Generic "skills" without qualifier — return all skill types
    (re.compile(r"\bskills?\b", re.IGNORECASE), "content:Skill"),
]


# Categories too broad for the comparative path — would dump hundreds of entries
# when a user asks "which item is best?" or "strongest mob?"
# These still work for explicit enumeration ("list all items?").
_BROAD_CATEGORIES: frozenset[str] = frozenset({
    "prefix:Items", "prefix:Items/Schematics", "prefix:Blocks",
    "prefix:Mobs", "prefix:Structures", "content:Skill",
})


def _detect_category(question: str, exclude_broad: bool = False) -> str | None:
    """Return the retrieval strategy if the question mentions a known category, or None."""
    for pattern, strategy in _CATEGORY_MAP:
        if pattern.search(question):
            if exclude_broad and strategy in _BROAD_CATEGORIES:
                continue
            return strategy
    return None


def _detect_enumeration(question: str) -> str | None:
    """If the question asks to list/enumerate a category, return the retrieval strategy.

    Returns a string like "prefix:Races" or "content:Unique Skill", or None.
    """
    # Must match an enumeration pattern first
    if not any(p.search(question) for p in _ENUM_PATTERNS):
        return None
    return _detect_category(question)


def _fetch_category_chunks(collection, strategy: str) -> list[dict]:
    """Fetch all summary chunks for a category, sorted by page title.

    prefix strategies: all pages whose title starts with the given path.
    content strategies: all summary chunks whose text contains the marker.
    """
    kind, value = strategy.split(":", 1)

    if kind == "prefix":
        result = collection.get(
            where={
                "$and": [
                    {"chunk_type": {"$eq": "summary"}},
                    {"page_title": {"$ne": value}},
                ]
            },
            include=["documents", "metadatas"],
        )
        chunks = [
            {"text": doc, "page_title": meta.get("page_title", ""),
             "section": meta.get("section", ""), "url": meta.get("url", ""), "score": 1.0}
            for doc, meta in zip(result["documents"], result["metadatas"])
            if meta.get("page_title", "").startswith(value + "/")
            and not _is_blocked(meta.get("page_title", ""))
        ]
    elif kind == "allcontent":
        # Search ALL chunks (not just summaries) for the marker — used for
        # sub-categories like Blessings/Curses within the Engravings page.
        result = collection.get(include=["documents", "metadatas"])
        value_lower = value.lower()
        chunks = [
            {"text": doc, "page_title": meta.get("page_title", ""),
             "section": meta.get("section", ""), "url": meta.get("url", ""), "score": 1.0}
            for doc, meta in zip(result["documents"], result["metadatas"])
            if value_lower in doc.lower() and not _is_blocked(meta.get("page_title", ""))
        ]
    else:
        # content: search summary chunks only
        result = collection.get(
            where={"chunk_type": {"$eq": "summary"}},
            include=["documents", "metadatas"],
        )
        chunks = [
            {"text": doc, "page_title": meta.get("page_title", ""),
             "section": meta.get("section", ""), "url": meta.get("url", ""), "score": 1.0}
            for doc, meta in zip(result["documents"], result["metadatas"])
            if value in doc and not _is_blocked(meta.get("page_title", ""))
        ]

    chunks.sort(key=lambda c: c["page_title"])
    return chunks


def _condense_chunks(chunks: list[dict], verbose: bool = False) -> list[dict]:
    """Condense category chunks to fit within token budget.

    verbose=False: light — name + key identifiers (for enumeration).
    verbose=True:  rich — full stats, intrinsics, evolution (for comparison).
    """
    keywords = _CONDENSATION_VERBOSE if verbose else _CONDENSATION_LIGHT
    char_budget = MAX_COMPARATIVE_CHARS if verbose else MAX_ENUM_CHARS

    condensed: list[dict] = []
    total_chars = 0
    for c in chunks:
        lines = c["text"].split("\n")
        kept = [lines[0]] if lines else []
        for line in lines[1:]:
            if any(kw in line for kw in keywords):
                kept.append(line)
        entry_text = "\n".join(kept)
        if total_chars + len(entry_text) > char_budget:
            logger.info("Category hit %d char budget at %d/%d entries", char_budget, len(condensed), len(chunks))
            break
        condensed.append({**c, "text": entry_text})
        total_chars += len(entry_text)

    logger.info("Category: %d/%d entries, %d chars (verbose=%s)", len(condensed), len(chunks), total_chars, verbose)
    return condensed


def _query_category(collection, strategy: str, verbose: bool = False) -> list[dict]:
    """Fetch and condense all summary chunks for a category."""
    chunks = _fetch_category_chunks(collection, strategy)
    return _condense_chunks(chunks, verbose=verbose)


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


# Version/changelog pages — broad vocabulary pollutes almost every query
_VERSION_PAGE_RE = re.compile(r"^\d+\.\d+")


def _is_blocked(page_title: str) -> bool:
    """Return True if a page title is on the blocklist."""
    if page_title in _EXACT_BLOCKED:
        return True
    if _VERSION_PAGE_RE.match(page_title):
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

# Player vocabulary → wiki vocabulary mappings.
# Only applied when the ENTIRE bare query matches a race-context term.
# Not applied globally because "Demon" is legitimate in non-race contexts
# (Demon Essence, Demon Lord Haki, Demon Marionette).
_RACE_SYNONYM_PHRASES: dict[str, str] = {
    "lesser demon": "lesser daemon",
    "greater demon": "greater daemon",
    "arch demon": "arch daemon",
    "demon lord": "daemon lord",
    "devil lord": "devil lord",
    "demon slime": "demon slime",
}


def _normalize_query(question: str) -> str:
    """Replace race-context phrases with wiki-canonical vocabulary.

    Only maps known multi-word race phrases (e.g. "lesser demon" → "lesser daemon").
    Leaves single-word "demon" untouched to avoid masking pages like Demon Essence.
    """
    lower = question.lower().rstrip("?!.,;: ")
    for player_term, wiki_term in _RACE_SYNONYM_PHRASES.items():
        if player_term in lower:
            # Case-preserving replacement
            idx = lower.find(player_term)
            original = question[idx:idx + len(player_term)]
            if original[0].isupper():
                replacement = wiki_term.title()
            else:
                replacement = wiki_term
            question = question[:idx] + replacement + question[idx + len(player_term):]
            lower = question.lower().rstrip("?!.,;: ")
    return question


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

    Tries the entity as-is, title-cased, and with all known wiki prefixes
    (e.g. "Alignment" → "Races/Alignment", "Inspire" → checked under all prefixes).
    Returns an empty list if no matching page is found or if the page is blocked.
    """
    titled = entity.title()
    candidates = [entity, titled]
    # Also try all known prefixes (Races/X, Mobs/X, Abilities/Magics/X, etc.)
    for prefix in _get_title_prefixes(collection):
        candidates.append(f"{prefix}{titled}")
    candidates = list(dict.fromkeys(candidates))  # deduplicate, preserve order

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
                    "score": 1.0,
                }
                for text, meta in zip(result["documents"], result["metadatas"])
            ]
    return []


_MAX_TITLE_BOOSTS = 3  # cap to avoid flooding variants on long queries

# Cache of all page title prefixes (e.g. "Races/", "Mobs/", "Abilities/Magics/")
# Built lazily on first use from the ChromaDB index.
_title_prefixes: list[str] | None = None
_title_prefixes_lock = threading.Lock()


def _get_title_prefixes(collection) -> list[str]:
    """Return all page title prefixes sorted longest-first for greedy matching."""
    global _title_prefixes
    if _title_prefixes is not None:
        return _title_prefixes
    with _title_prefixes_lock:
        if _title_prefixes is not None:
            return _title_prefixes
        result = collection.get(include=["metadatas"])
        prefixes: set[str] = set()
        for meta in result["metadatas"]:
            title = meta.get("page_title", "")
            if "/" in title:
                prefixes.add(title.rsplit("/", 1)[0] + "/")
        _title_prefixes = sorted(prefixes, key=len, reverse=True)
        logger.debug("Loaded %d page title prefixes", len(_title_prefixes))
    return _title_prefixes


def _inject_page_title_variant(
    collection, question: str, variants: list[str],
) -> None:
    """If words in the question match page titles, add them as search variants.

    Prevents common terms ("cooldown", "damage") from drowning entity names
    ("creator", "predator") in the embedding. Checks 1-, 2-, and 3-word windows
    from the question against ChromaDB page titles.

    For multi-entity queries ("hero as a giant?"), injects ALL matching page
    titles (up to _MAX_TITLE_BOOSTS) so both entities get retrieval coverage.
    """
    words = question.rstrip("?").strip().split()
    # Skip question-starter words for candidate generation
    content_words = [w for w in words if w.lower() not in _QUESTION_STARTERS]
    if not content_words:
        return

    # Build candidate phrases: trigrams first (more specific), then bigrams, then unigrams
    candidates: list[str] = []
    for size in (3, 2, 1):
        for i in range(len(content_words) - size + 1):
            phrase = " ".join(content_words[i:i + size])
            if len(phrase) >= 3:  # skip very short candidates
                candidates.append(phrase)

    boosts = 0
    matched_words: set[str] = set()  # track which words already matched to avoid subset dupes
    for phrase in candidates:
        if boosts >= _MAX_TITLE_BOOSTS:
            break
        if phrase in variants:
            continue
        # Skip if all words in this phrase were already matched by a longer phrase
        phrase_words = set(phrase.lower().split())
        if phrase_words <= matched_words:
            continue
        # Try exact page title match, then every known prefix (Races/X, Mobs/X, etc.)
        titled = phrase.title()
        forms = [phrase, titled]
        for prefix in _get_title_prefixes(collection):
            forms.append(f"{prefix}{titled}")
        # Deduplicate while preserving order
        forms = list(dict.fromkeys(forms))
        for form in forms:
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
                matched_words.update(phrase_words)
                boosts += 1
                break


def _semantic_search(collection, variants: list[str], k: int) -> list[dict]:
    """Run each query variant through ChromaDB and return top-K merged results.

    Deduplicates by chunk text, keeping the highest score per unique chunk.
    """
    embedder = _get_embedder()
    seen: dict[str, dict] = {}

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
            score = 1.0 - (dist / 2.0)
            if score < RELEVANCE_THRESHOLD:
                continue
            page_title = meta.get("page_title", "")
            if _is_blocked(page_title):
                continue
            if text not in seen or score > seen[text]["score"]:
                seen[text] = {
                    "text": text,
                    "page_title": page_title,
                    "section": meta.get("section", ""),
                    "url": meta.get("url", ""),
                    "score": round(score, 3),
                }

    return sorted(seen.values(), key=lambda c: c["score"], reverse=True)[:k]


def query(question: str) -> list[dict]:
    """Embed the question, search ChromaDB, and return the most relevant chunks.

    Routing order: enumeration/comparative → bare entity → semantic search.
    Each returned dict has keys: text, page_title, section, url, score.
    """
    collection = get_collection()
    question = _normalize_query(question)

    # Fast path: enumeration or comparative + category
    enum_strategy = _detect_enumeration(question)
    comparative = is_comparative(question)
    if not enum_strategy and comparative:
        enum_strategy = _detect_category(question, exclude_broad=True)
    if enum_strategy:
        enum_chunks = _query_category(collection, enum_strategy, verbose=comparative)
        if enum_chunks:
            return enum_chunks

    # Fast path: bare entity lookup (e.g. "Predator?", "Great Sage?")
    entity = _bare_entity_name(question)
    if entity:
        page_chunks = _query_by_page_title(collection, entity)
        if page_chunks:
            return page_chunks
        # Bare entity failed — check if the ENTIRE entity is a category name
        # (e.g. "Blessings?" → yes, "Magic Ore?" → no, "Anti skill?" → no).
        # Only triggers when the entity has 1 word matching a category keyword.
        if len(entity.split()) == 1:
            cat = _detect_category(question)
            if cat:
                cat_chunks = _query_category(collection, cat)
                if cat_chunks:
                    return cat_chunks

    # Semantic search with query expansion and page title boosting
    k = K_COMPARATIVE if comparative else K_FACTUAL
    variants = [question] + expand_query(question)
    if entity and entity not in variants:
        variants.append(entity)
    _inject_page_title_variant(collection, question, variants)
    logger.debug("Query variants (%d): %s", len(variants), variants)

    results = _semantic_search(collection, variants, k)

    # Last resort: if semantic search returned nothing and we have a short entity,
    # do a text-contains search across all chunks (catches sub-abilities like
    # "Inspire" within Commander, or "Hero's Blessing" within Chosen One).
    if not results and entity and len(entity) >= 4:
        logger.debug("Semantic search empty for %r, trying text-contains fallback", entity)
        all_chunks = collection.get(include=["documents", "metadatas"])
        entity_lower = entity.lower()
        text_matches = [
            {"text": doc, "page_title": meta.get("page_title", ""),
             "section": meta.get("section", ""), "url": meta.get("url", ""), "score": 0.5}
            for doc, meta in zip(all_chunks["documents"], all_chunks["metadatas"])
            if entity_lower in doc.lower() and not _is_blocked(meta.get("page_title", ""))
        ]
        if text_matches:
            results = sorted(text_matches, key=lambda c: c["score"], reverse=True)[:k]
            logger.info("Text-contains fallback: %d chunks for %r", len(results), entity)

    return results
