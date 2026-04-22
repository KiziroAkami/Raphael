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


# Compound phrases where a category keyword appears as a noun modifier, not as
# a category indicator. If a question contains the phrase, the associated
# strategies are suppressed so the retriever does not route to the wrong
# category. (TEN-187.) Add new entries as specific mis-routings are found.
_COMPOUND_EXCLUSIONS: list[tuple[re.Pattern[str], frozenset[str]]] = [
    # "magic ore" is a crafting material, not a spell. The "magic" substring
    # would otherwise trigger the Abilities/Magics prefix category.
    (
        re.compile(r"\bmagic\s+ores?\b", re.IGNORECASE),
        frozenset({"prefix:Abilities/Magics"}),
    ),
]


def _suppressed_strategies(question: str) -> set[str]:
    """Return the set of category strategies blocked by compound-term exclusions."""
    suppressed: set[str] = set()
    for pattern, strategies in _COMPOUND_EXCLUSIONS:
        if pattern.search(question):
            suppressed |= strategies
    return suppressed


def _detect_category(question: str, exclude_broad: bool = False) -> str | None:
    """Return the retrieval strategy if the question mentions a known category, or None."""
    suppressed = _suppressed_strategies(question)
    for pattern, strategy in _CATEGORY_MAP:
        if pattern.search(question):
            if strategy in suppressed:
                continue
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
            if meta is not None
            and meta.get("page_title", "").startswith(value + "/")
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
            if meta is not None
            and value_lower in doc.lower()
            and not _is_blocked(meta.get("page_title", ""))
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
            if meta is not None
            and value in doc
            and not _is_blocked(meta.get("page_title", ""))
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


# Maps strategy prefixes to the human noun the LLM should mention when it
# tells users the list was truncated. (TEN-190.)
_STRATEGY_NOUN: dict[str, str] = {
    "prefix:Races": "race",
    "prefix:Mobs": "mob",
    "prefix:Abilities/Magics": "spell",
    "prefix:Abilities/Battlewills": "battlewill",
    "prefix:Effects": "effect",
    "prefix:Blocks": "block",
    "prefix:Structures": "structure",
    "prefix:Items/Schematics": "schematic",
    "prefix:Items": "item",
    "content:Unique Skill": "unique skill",
    "content:Extra Skill": "extra skill",
    "content:Common Skill": "common skill",
    "content:Intrinsic Skill": "intrinsic skill",
    "content:Resistance Skill": "resistance skill",
    "content:Skill": "skill",
    "allcontent:Blessing": "blessing",
    "allcontent:Curse": "curse",
    "allcontent:Engravings —": "engraving",
}


def _append_truncation_notice(
    condensed: list[dict], total: int, strategy: str,
) -> list[dict]:
    """If condensation dropped entries, append an in-context truncation note.

    The note is appended to the LAST kept chunk (immutably — no mutation) so
    the LLM sees it inline with the wiki content and can mention the partial
    nature of the list in its reply. Uses ``[Note: ...]`` as a distinctive
    marker that won't be confused with wiki text. (TEN-190.)
    """
    dropped = total - len(condensed)
    if dropped <= 0 or not condensed:
        return condensed
    noun = _STRATEGY_NOUN.get(strategy, "entry")
    plural = noun + ("es" if noun.endswith(("s", "x", "ch", "sh")) else "s")
    note = (
        f"\n\n[Note: {dropped} additional {plural} omitted for length. "
        f"Ask about a specific {noun} for full details.]"
    )
    last = condensed[-1]
    return condensed[:-1] + [{**last, "text": last["text"] + note}]


def _query_category(collection, strategy: str, verbose: bool = False) -> list[dict]:
    """Fetch and condense all summary chunks for a category."""
    chunks = _fetch_category_chunks(collection, strategy)
    condensed = _condense_chunks(chunks, verbose=verbose)
    return _append_truncation_notice(condensed, total=len(chunks), strategy=strategy)


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

# Compound page names that contain a race-synonym phrase as a prefix but refer
# to a different entity (skill, item, etc.).  Normalisation is skipped when any
# of these appear in the query so "Demon Lord Haki" stays intact.
_NORMALIZE_EXCLUDE: frozenset[str] = frozenset({
    "demon lord haki",
    "demon lord-level",
    "demon lord level",
})


def _normalize_query(question: str) -> str:
    """Replace race-context phrases with wiki-canonical vocabulary.

    Only maps known multi-word race phrases (e.g. "lesser demon" → "lesser daemon").
    Leaves single-word "demon" untouched to avoid masking pages like Demon Essence.
    Skips normalisation when the phrase is part of a compound entity name listed
    in ``_NORMALIZE_EXCLUDE`` (e.g. "Demon Lord Haki").
    """
    lower = question.lower().rstrip("?!.,;: ")
    for player_term, wiki_term in _RACE_SYNONYM_PHRASES.items():
        if player_term in lower:
            # Skip if the matched phrase is part of a longer compound term
            if any(excl in lower for excl in _NORMALIZE_EXCLUDE if player_term in excl):
                continue
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


def _fetch_page_chunks(collection, page_title: str, score: float) -> list[dict]:
    """Return all chunks for a given page_title as result dicts, or [] if not found."""
    result = collection.get(
        where={"page_title": {"$eq": page_title}},
        include=["documents", "metadatas"],
    )
    if not result["documents"]:
        return []
    return [
        {
            "text": text,
            "page_title": meta.get("page_title", ""),
            "section": meta.get("section", ""),
            "url": meta.get("url", ""),
            "score": score,
        }
        for text, meta in zip(result["documents"], result["metadatas"])
        if meta is not None
    ]


def _query_by_page_title(collection, entity: str) -> list[dict]:
    """Return all chunks for a wiki page matching the entity name.

    First tries exact matches: the entity as-is, title-cased, and under each
    known wiki prefix (e.g. ``Alignment`` → ``Races/Alignment``).

    If no exact match, consults the word-prefix index so that partial lookups
    like ``hinata`` surface ``Mobs/Hinata Sakaguchi``. Returns the chunks of
    the shortest matching page (most specific). (TEN-198.)

    Returns an empty list if no matching page is found or if the page is blocked.
    """
    titled = entity.title()
    candidates = [entity, titled]
    for prefix in _get_title_prefixes(collection):
        candidates.append(f"{prefix}{titled}")
    candidates = list(dict.fromkeys(candidates))  # deduplicate, preserve order

    for candidate in candidates:
        if _is_blocked(candidate):
            logger.debug("Bare entity %r is on blocklist, skipping", candidate)
            continue
        chunks = _fetch_page_chunks(collection, candidate, score=1.0)
        if chunks:
            logger.debug("Bare entity hit: page_title=%r (%d chunks)", candidate, len(chunks))
            return chunks

    # Word-prefix fallback: "hinata" matches "Mobs/Hinata Sakaguchi".
    # Only commit to a match when it's unambiguous — otherwise "fire?" (8
    # candidates) would arbitrarily snap to "Fire Ball". When ambiguous, fall
    # through to semantic search where ranking can disambiguate by context.
    idx = _get_title_word_prefix_index(collection)
    unblocked = [t for t in idx.get(entity.lower(), []) if not _is_blocked(t)]
    if len(unblocked) == 1:
        chunks = _fetch_page_chunks(collection, unblocked[0], score=1.0)
        if chunks:
            logger.debug("Bare entity prefix-hit: %r → %r (%d chunks)", entity, unblocked[0], len(chunks))
            return chunks
    return []


_MAX_TITLE_BOOSTS = 3  # cap to avoid flooding variants on long queries
_TITLE_COMPONENT_MAX_WORDS = 4  # n-gram width for the word-prefix index

# Cache of all page title prefixes (e.g. "Races/", "Mobs/", "Abilities/Magics/")
# Built lazily on first use from the ChromaDB index.
_title_prefixes: list[str] | None = None
_title_word_prefix_index: dict[str, list[str]] | None = None
_title_prefixes_lock = threading.Lock()


def _get_title_prefixes(collection) -> list[str]:
    """Return all page title prefixes sorted longest-first for greedy matching."""
    global _title_prefixes
    if _title_prefixes is not None:
        return _title_prefixes
    with _title_prefixes_lock:
        if _title_prefixes is not None:
            return _title_prefixes
        _build_title_caches(collection)
        assert _title_prefixes is not None
    return _title_prefixes


def _get_title_word_prefix_index(collection) -> dict[str, list[str]]:
    """Return {lowercase-word-prefix: [matching_page_titles]} index.

    The last path component of each page title contributes 1..N word-prefixes:
    ``Mobs/Hinata Sakaguchi`` contributes keys ``"hinata"`` and ``"hinata sakaguchi"``.
    ``Chosen One`` contributes keys ``"chosen"`` and ``"chosen one"``.

    Matching values are deduplicated and sorted by path length so the most
    specific (shortest) match appears first. This powers both bare-entity
    lookup (TEN-198) and structured-question page boosts (TEN-201).
    """
    global _title_word_prefix_index
    if _title_word_prefix_index is not None:
        return _title_word_prefix_index
    with _title_prefixes_lock:
        if _title_word_prefix_index is not None:
            return _title_word_prefix_index
        _build_title_caches(collection)
        assert _title_word_prefix_index is not None
    return _title_word_prefix_index


def _build_title_caches(collection) -> None:
    """Populate ``_title_prefixes`` and ``_title_word_prefix_index`` in one pass."""
    global _title_prefixes, _title_word_prefix_index
    result = collection.get(include=["metadatas"])
    prefixes: set[str] = set()
    word_index: dict[str, set[str]] = {}
    for meta in result["metadatas"]:
        if meta is None:
            continue
        title = meta.get("page_title", "")
        if not title:
            continue
        if "/" in title:
            prefixes.add(title.rsplit("/", 1)[0] + "/")
        last = title.rsplit("/", 1)[-1]
        words = last.split()
        limit = min(len(words), _TITLE_COMPONENT_MAX_WORDS)
        for n in range(1, limit + 1):
            key = " ".join(words[:n]).lower()
            word_index.setdefault(key, set()).add(title)
    _title_prefixes = sorted(prefixes, key=len, reverse=True)
    _title_word_prefix_index = {
        k: sorted(v, key=lambda t: (len(t), t))
        for k, v in word_index.items()
    }
    logger.debug(
        "Built title caches: %d prefixes, %d word-prefix keys",
        len(_title_prefixes), len(_title_word_prefix_index),
    )


def _find_title_hits_in_question(collection, question: str) -> list[str]:
    """Return page titles whose last-component word-prefix matches phrases in the question.

    Scans n-grams of length 3..1 (longest first, most specific). For each phrase
    that matches the word-prefix index, picks the shortest matching page title
    (most specific). Caps at ``_MAX_TITLE_BOOSTS`` to avoid flooding variants on
    long queries. Skips phrases already covered by a longer matching phrase.

    Used by ``query()`` to directly inject matching page chunks (TEN-201) — a
    successor to the older "title as semantic variant" approach, which failed
    for short generic titles like ``Chosen One`` that embed to noise.
    """
    words = question.rstrip("?").strip().split()
    content_words = [w for w in words if w.lower() not in _QUESTION_STARTERS]
    if not content_words:
        return []

    idx = _get_title_word_prefix_index(collection)
    hits: list[str] = []
    matched_words: set[str] = set()

    for size in (3, 2, 1):
        if len(hits) >= _MAX_TITLE_BOOSTS:
            break
        for i in range(len(content_words) - size + 1):
            if len(hits) >= _MAX_TITLE_BOOSTS:
                break
            phrase = " ".join(content_words[i:i + size]).lower()
            if len(phrase) < 3:
                continue
            phrase_words = set(phrase.split())
            if phrase_words <= matched_words:
                continue
            # Require an unambiguous match: single-word phrases like "fire"
            # often hit 8+ pages and shouldn't arbitrarily snap to one.
            # Multi-word phrases like "chosen one" usually resolve uniquely.
            candidates = [t for t in idx.get(phrase, []) if not _is_blocked(t) and t not in hits]
            if len(candidates) != 1:
                continue
            chosen = candidates[0]
            hits.append(chosen)
            matched_words.update(phrase_words)
            logger.debug("Title boost: phrase %r → page %r", phrase, chosen)
    return hits


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
            if meta is None:
                continue
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


_MIN_TEXT_CONTAINS_ENTITY_LEN = 4

# Action verbs to strip when falling back to single-token text-contains on a
# multi-word entity (TEN-203). "Spawn gazel Dwargo" → drop "spawn", search "Gazel"/"Dwargo".
_ENTITY_ACTION_VERBS: frozenset[str] = frozenset({
    "spawn", "summon", "get", "make", "find", "kill", "give", "craft", "build",
})


def _text_contains_search(collection, entity: str, k: int, score: float) -> list[dict]:
    """Word-boundary text scan across all chunks for an entity name.

    Catches sub-abilities and character mentions that live inside parent pages
    (e.g. "Gazel" inside Mobs/Dwarf, "Inspire" inside Commander). Uses a
    ``\\b<entity>\\b`` regex so short tokens like "quest" don't match
    "request"/"conquest". (TEN-189.)
    """
    if len(entity) < _MIN_TEXT_CONTAINS_ENTITY_LEN:
        return []
    pattern = re.compile(rf"\b{re.escape(entity)}\b", re.IGNORECASE)
    all_chunks = collection.get(include=["documents", "metadatas"])
    matches = [
        {"text": doc, "page_title": meta.get("page_title", ""),
         "section": meta.get("section", ""), "url": meta.get("url", ""), "score": score}
        for doc, meta in zip(all_chunks["documents"], all_chunks["metadatas"])
        if meta is not None
        and pattern.search(doc)
        and not _is_blocked(meta.get("page_title", ""))
    ]
    if not matches:
        return []
    return sorted(matches, key=lambda c: c["score"], reverse=True)[:k]


def _single_token_fallback(collection, entity: str, k: int) -> list[dict]:
    """Retry text-contains on each content word of a multi-word entity.

    When the full multi-word phrase has zero matches, a single token often
    does: ``Spawn gazel Dwargo`` → ``Gazel`` inside ``Mobs/Dwarf``. Drops
    stopwords, action verbs, and tokens shorter than the text-contains
    minimum. Scores slightly below full-phrase (0.85 vs 0.9) because a
    single-token match is less specific. (TEN-203.)
    """
    tokens = [
        t for t in entity.split()
        if t.lower() not in _QUESTION_STARTERS
        and t.lower() not in _ENTITY_ACTION_VERBS
        and len(t) >= _MIN_TEXT_CONTAINS_ENTITY_LEN
    ]
    tokens.sort(key=len, reverse=True)  # longest first
    for tok in tokens:
        hits = _text_contains_search(collection, tok, k=k, score=0.85)
        if hits:
            logger.info("Single-token fallback: %r → %d chunks via %r", entity, len(hits), tok)
            return hits
    return []


def _merge_results(direct: list[dict], semantic: list[dict], k: int) -> list[dict]:
    """Merge direct-page hits and semantic hits, dedup by chunk text, sort by score."""
    seen: dict[str, dict] = {}
    for c in list(direct) + list(semantic):
        text = c["text"]
        if text not in seen or c["score"] > seen[text]["score"]:
            seen[text] = c
    return sorted(seen.values(), key=lambda c: c["score"], reverse=True)[:k]


def query(question: str) -> list[dict]:
    """Embed the question, search ChromaDB, and return the most relevant chunks.

    Routing order: enumeration/comparative → bare entity → direct page-title
    hits + semantic → text-contains fallbacks. Each returned dict has keys:
    text, page_title, section, url, score.
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
        # Page title + category both missed. Try a text-contains search BEFORE
        # semantic, so entity mentions living inside other pages (e.g. "Gazel"
        # inside Mobs/Dwarf) aren't masked by noisy weak semantic hits. (TEN-189.)
        text_hits = _text_contains_search(collection, entity, k=K_FACTUAL, score=0.9)
        if text_hits:
            logger.info("Text-contains (bare entity): %d chunks for %r", len(text_hits), entity)
            return text_hits
        # Full phrase missed. For multi-word entities, retry on individual
        # tokens so "Gazel Dwargo" / "Spawn gazel Dwargo" surface Mobs/Dwarf
        # mentions. (TEN-203.)
        if " " in entity:
            token_hits = _single_token_fallback(collection, entity, k=K_FACTUAL)
            if token_hits:
                return token_hits

    # Direct page-title hits for structured queries (TEN-201).
    # Finds pages whose titles appear in the question and fetches their chunks
    # at score 0.95 — higher than typical semantic hits, so they rise to the
    # top even when the page's title embeds poorly (e.g. "Chosen One").
    k = K_COMPARATIVE if comparative else K_FACTUAL
    direct_chunks: list[dict] = []
    for page_title in _find_title_hits_in_question(collection, question):
        direct_chunks.extend(_fetch_page_chunks(collection, page_title, score=0.95))

    # Semantic search with query expansion
    variants = [question] + expand_query(question)
    if entity and entity not in variants:
        variants.append(entity)
    logger.debug("Query variants (%d): %s", len(variants), variants)

    semantic_chunks = _semantic_search(collection, variants, k)

    results = _merge_results(direct_chunks, semantic_chunks, k) if direct_chunks else semantic_chunks

    # Last resort: if search returned nothing and we have a short entity,
    # do a text-contains search across all chunks.
    if not results and entity:
        text_hits = _text_contains_search(collection, entity, k=k, score=0.5)
        if text_hits:
            logger.info("Text-contains fallback: %d chunks for %r", len(text_hits), entity)
            results = text_hits

    return results
