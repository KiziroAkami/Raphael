import hashlib
import logging
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma"
INDEX_VERSION_FILE = Path(__file__).parent.parent / "data" / ".index_version"
logger = logging.getLogger(__name__)
COLLECTION_NAME = "tensura_wiki"
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_CHUNK_TOKENS = 400   # ~1,600 chars; hard cap per chunk
CHARS_PER_TOKEN = 4      # rough estimate for splitting
MAX_CHUNK_CHARS = MAX_CHUNK_TOKENS * CHARS_PER_TOKEN
MIN_CHUNK_CHARS = 20     # skip near-empty chunks (stubs, placeholders, leaked headings)

# Templates whose key=value pairs contain game stats worth indexing.
# These are converted to "Key: Value\n..." text instead of being stripped.
DATA_TEMPLATES = re.compile(
    r"(?:infobox|creature|skill|magic|race|item|effect)\b",
    re.IGNORECASE,
)

# Navbox/layout templates to strip entirely (case-insensitive prefix match)
LAYOUT_TEMPLATE_PREFIXES = re.compile(
    r"(?:navbox|toc|wip|displaytitle|historytable|historyline|"
    r"skillsnavbox|magicsnavbox|mobsnavbox)\b",
    re.IGNORECASE,
)

_embedder: SentenceTransformer | None = None
_chroma_client: chromadb.PersistentClient | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model...")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_collection() -> chromadb.Collection:
    return _get_chroma_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> None:
    """Drop and recreate the collection — use before a full rebuild to clear stale chunks."""
    client = _get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        pass
    client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    touch_index_version()


def touch_index_version() -> None:
    """Bump the index version mtime so running bot processes know to invalidate
    their module-level title caches. The retriever checks this file's mtime
    before each cache-backed query. Cross-process safe because the sync cron
    runs in a separate process and can't invalidate the bot's in-memory globals
    directly."""
    INDEX_VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_VERSION_FILE.touch()


# ---------------------------------------------------------------------------
# Wikitext cleaning
# ---------------------------------------------------------------------------

_WIKI_LINK_RE = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
_IMAGE_KEY_RE = re.compile(r"^images?$", re.IGNORECASE)


def _expand_data_template(inner: str) -> str:
    """Convert template key=value pairs to readable 'Key: Value' lines.

    Skips image/filename fields and empty values so they don't pollute chunks.
    Wiki links inside values are resolved to their display text before
    splitting on `|`, so [[Page|Display]] doesn't confuse the field parser.
    """
    # Resolve wiki links first so their internal | doesn't split fields
    resolved = _WIKI_LINK_RE.sub(r"\1", inner)
    pairs = re.findall(r"\|\s*([^=|{}\n]+?)\s*=\s*([^|{}\n]+)", resolved)
    lines = []
    for k, v in pairs:
        key = k.strip()
        val = v.strip()
        if not key or not val:
            continue
        if _IMAGE_KEY_RE.match(key):
            continue
        lines.append(f"{key}: {val}")
    return "\n".join(lines)


def _clean_wikitext(raw: str) -> str:
    """Strip wikitext noise while preserving stats and readable content."""
    text = raw

    # Remove ref tags and their content
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/?>", "", text)

    # Remove HTML comments and tags
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)

    # DescriptionEcho: {{DescriptionEcho|Some plain text.}} → the plain text
    text = re.sub(r"\{\{DescriptionEcho\|([^}]+)\}\}", r"\1", text, flags=re.IGNORECASE)

    # ItemLink: {{ItemLink|Item Name}} → Item Name
    text = re.sub(r"\{\{ItemLink\|([^|}]+)(?:\|[^}]*)?\}\}", r"\1", text, flags=re.IGNORECASE)

    # Expand data templates (Infobox, Creature, Skill, Magic, Race, Item …)
    # to readable "Key: Value" lines
    def _maybe_expand(m: re.Match) -> str:
        template_name = m.group(1)
        if DATA_TEMPLATES.match(template_name):
            return _expand_data_template(m.group(0))
        if LAYOUT_TEMPLATE_PREFIXES.match(template_name):
            return ""
        return m.group(0)  # leave unknown templates for the sweep below

    text = re.sub(
        r"\{\{([A-Za-z][^|{}\n]*?)\s*[\n|]((?:[^{}]|\{\{[^{}]*\}\})*)\}\}",
        _maybe_expand,
        text,
        flags=re.DOTALL,
    )

    # Convert wikitables: extract cell values
    text = re.sub(r"^\s*[|!]{1,2}", "", text, flags=re.MULTILINE)  # strip | and !! row starters
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)        # remove remaining table wrappers

    # Remove remaining templates ({{...}}) that weren't caught above
    # Do multiple passes for nested templates
    for _ in range(3):
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)

    # Strip wiki links but keep display text: [[Page|Display]] → Display, [[Page]] → Page
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)

    # Strip external links: [http://... text] → text, bare URLs → removed
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)
    text = re.sub(r"https?://\S+", "", text)  # bare URLs (e.g. Tutorial Video sections)

    # Strip remaining wiki markup
    text = re.sub(r"'{2,3}", "", text)   # bold/italic
    text = re.sub(r"={2,6}(.+?)={2,6}", r"\1", text)  # headings → plain text
    text = re.sub(r"^[\s*#:;]+", "", text, flags=re.MULTILINE)  # list markers

    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_sections(wikitext: str) -> list[tuple[str, str]]:
    """Split wikitext into (section_name, raw_text) pairs on == headings."""
    pattern = re.compile(r"^={2,4}\s*(.+?)\s*={2,4}", re.MULTILINE)
    sections = []
    pos = 0
    current_section = "Overview"

    for match in pattern.finditer(wikitext):
        chunk = wikitext[pos:match.start()].strip()
        if chunk:
            sections.append((current_section, chunk))
        current_section = match.group(1).strip()
        pos = match.end()

    # remainder after last heading
    remainder = wikitext[pos:].strip()
    if remainder:
        sections.append((current_section, remainder))

    return sections


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """Split text that exceeds max_chars at paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            # paragraph itself too long: hard-split
            if len(para) > max_chars:
                for i in range(0, len(para), max_chars):
                    chunks.append(para[i:i + max_chars])
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


def _extract_infobox_stats(wikitext: str) -> str:
    """Pull key=value pairs from the first data template for the summary chunk."""
    m = re.search(
        r"\{\{([A-Za-z][^|{}\n]*?)\s*[\n|]((?:[^{}]|\{\{[^{}]*\}\})*)\}\}",
        wikitext,
        re.DOTALL,
    )
    if not m or not DATA_TEMPLATES.match(m.group(1)):
        return ""
    return _expand_data_template(m.group(0))


def build_chunks(page: dict) -> list[dict]:
    """Convert a raw wiki page into a list of chunk dicts ready for embedding."""
    missing = [k for k in ("title", "url", "wikitext") if not page.get(k)]
    if missing:
        raise ValueError(f"Page dict missing required keys: {missing}")
    title = page["title"]
    url = page["url"]
    wikitext = page["wikitext"]

    # Skip redirect pages — they produce useless "#REDIRECT [[Target]]" chunks
    # that crowd out real content in retrieval results
    if wikitext.strip().upper().startswith("#REDIRECT"):
        return []

    chunks = []

    # 1. Summary chunk: stats OR first paragraph (not both — avoids duplicating
    #    the same template block that the section chunks already cover)
    stats = _extract_infobox_stats(wikitext)
    clean_full = _clean_wikitext(wikitext)
    first_para = clean_full.split("\n\n")[0] if clean_full else ""
    summary_text = (f"{title}\n{stats}" if stats else f"{title}\n{first_para}").strip()
    if len(summary_text) >= MIN_CHUNK_CHARS:
        chunks.append({
            "text": summary_text[:MAX_CHUNK_CHARS],
            "page_title": title,
            "section": "_summary",
            "url": url,
            "chunk_type": "summary",
        })

    # 2. Section chunks
    # Context prefix is prepended to every section chunk so that short sections
    # (e.g. "0 - 2 Leather\n0 - 2 Bones") embed near relevant queries
    # ("what does Goblin drop?") rather than floating in unrelated vector space.
    for section_name, raw_section in _split_sections(wikitext):
        clean = _clean_wikitext(raw_section)
        if len(clean) < MIN_CHUNK_CHARS:
            continue
        context_prefix = f"{title} — {section_name}:\n"
        max_body = MAX_CHUNK_CHARS - len(context_prefix)
        for part in _split_long_text(clean, max_body):
            if len(part) >= MIN_CHUNK_CHARS:
                chunks.append({
                    "text": context_prefix + part,
                    "page_title": title,
                    "section": section_name,
                    "url": url,
                    "chunk_type": "section",
                })

    return chunks


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def delete_page_chunks(page_title: str) -> None:
    """Remove all stored chunks for a single wiki page from ChromaDB."""
    collection = get_collection()
    collection.delete(where={"page_title": page_title})
    touch_index_version()


def reindex_page(page: dict) -> None:
    """Delete a page's old chunks and re-index the new version.

    Used by incremental updates — only touches the pages that changed.
    """
    delete_page_chunks(page["title"])
    build_index([page])


def build_index(pages: list[dict], batch_size: int = 64) -> None:
    """Embed all chunks and upsert into ChromaDB."""
    collection = get_collection()
    embedder = _get_embedder()

    all_chunks: list[dict] = []
    for page in pages:
        try:
            all_chunks.extend(build_chunks(page))
        except (ValueError, KeyError) as exc:
            logger.warning("Skipping page %r: %s", page.get("title", "?"), exc)

    logger.info("Total chunks to index: %d", len(all_chunks))

    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start:start + batch_size]
        texts = [c["text"] for c in batch]
        ids = [
            hashlib.sha256(
                f"{c['page_title']}|{c['section']}|{c['chunk_type']}|{c['text']}".encode()
            ).hexdigest()
            for c in batch
        ]
        metadatas = [
            {
                "page_title": c["page_title"],
                "section": c["section"],
                "url": c["url"],
                "chunk_type": c["chunk_type"],
            }
            for c in batch
        ]
        embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
        collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        logger.info("Indexed %d/%d", min(start + batch_size, len(all_chunks)), len(all_chunks))

    if all_chunks:
        touch_index_version()
    logger.info("Index build complete.")
