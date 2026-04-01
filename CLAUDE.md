# CLAUDE.md — Raphael

## Project Overview

**Raphael** is a Discord bot themed as Raphael, Lord of Wisdom from the Tensura anime. It answers questions about the [Tensura Minecraft mod](https://tensura.wiki.gg/) by retrieving wiki data and responding in-character.

**Trigger:** Any Discord message ending with `?` triggers a bot response (max 500 chars).

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Discord bot | `discord.py >= 2.3.0` |
| LLM inference | Groq API (`llama-3.3-70b-versatile` → fallback `qwen/qwen3-32b`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, local, no API) |
| Vector store | ChromaDB (persistent, local at `data/chroma/`) |
| Wiki scraping | `mwclient` (MediaWiki API for `tensura.wiki.gg`) |
| Process management | PM2 (Node.js-based, keeps bot running + daily sync cron) |
| macOS sleep prevention | `caffeinate -di` (via `scripts/start.sh`) |
| Python version | 3.14 (`.venv/`) |

---

## Directory Structure

```
bot/        Discord event handlers (on_ready, on_message, response chunker)
llm/        Groq API wrapper + Raphael persona prompts
rag/        Wiki scraper, wikitext cleaner/chunker, ChromaDB indexer, semantic retriever
scripts/    build_index.py (CLI for scraping/indexing), start.sh (caffeinate wrapper)
data/       Runtime data — NOT in git
  pages/    Cached wiki pages as JSON (~1279 files)
  chroma/   ChromaDB vector store (binary)
  last_updated.txt  ISO timestamp of last sync
main.py     Entry point — logging setup, Discord client init
ecosystem.config.cjs  PM2 config (bot process + daily sync cron at 03:00)
```

---

## Environment Variables

Both are required — startup raises `RuntimeError` if missing.

```
DISCORD_TOKEN=   # Discord bot token
GROQ_API_KEY=    # Groq API key
```

Copy `.env.example` → `.env` and fill in values.

---

## How to Run

### First-time setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in .env
python scripts/build_index.py        # Full scrape + index (~7 min, ~836 pages)
npm install -g pm2
pm2 start ecosystem.config.cjs
pm2 save && pm2 startup
```

### Daily usage
```bash
pm2 status                           # Check bot + sync processes
pm2 logs raphael                     # Tail bot logs
pm2 restart raphael                  # After code changes
```

### Index management
```bash
python scripts/build_index.py                # Full scrape + build
python scripts/build_index.py --incremental  # Only changed pages (runs automatically at 03:00)
python scripts/build_index.py --cached       # Re-index from disk cache (no API calls)
python scripts/build_index.py --refresh      # Force re-fetch all pages
```

### Verify index
```bash
python -c "from rag.indexer import get_collection; c = get_collection(); print(len(c.get()['ids']), 'chunks indexed')"
```

---

## Architecture

```
Discord message ("...?")
    └── bot/events.py (on_message)
            ├── rag/retriever.py  →  ChromaDB (data/chroma/)
            │       └── rag/indexer.py  ←  rag/scraper.py  ←  tensura.wiki.gg
            └── llm/client.py  (Groq API)
                    └── llm/prompts.py  (Raphael persona + RAG prompt)
```

**PM2 processes** (`ecosystem.config.cjs`):

```
pm2
├── raphael              .venv/bin/python main.py        (Discord bot — always running)
├── raphael-caffeinate   bash scripts/start.sh           (macOS sleep prevention — always running)
└── raphael-sync         .venv/bin/python build_index.py  (daily wiki sync at 03:00 — one-shot)
```

---

## Code Conventions

**Logging**
- All modules: `logging.getLogger(__name__)` — never `print()` (except `scripts/build_index.py` CLI)
- Format: `%(asctime)s %(levelname)s %(name)s: %(message)s`

**Error handling**
- Graceful degradation: primary model → fallback model → in-character error message
- Rate limit backoff: exponential (30s / 60s / 120s)
- User-facing errors stay in-character: _"Insufficient data in Raphael's archives"_

**Type hints**
- All function signatures use type annotations with explicit return types

**Paths**
- Always `pathlib.Path`, resolved relative to `Path(__file__).parent`

**Async**
- Discord handlers are `async`; RAG + LLM calls run in thread pool via `run_in_executor()`

**Immutability**
- Return new objects; don't mutate state in-place

**File size**
- Keep files under 800 lines, functions under 50 lines

---

## Constants (don't hardcode inline)

Key values live at module level in their respective files:

| Constant | Location | Value |
|---------|----------|-------|
| `PRIMARY_MODEL` | `llm/client.py` | `llama-3.3-70b-versatile` |
| `FALLBACK_MODEL` | `llm/client.py` | `qwen/qwen3-32b` |
| `MAX_TOKENS` | `llm/client.py` | 1024 |
| `TEMPERATURE` | `llm/client.py` | 0.3 |
| `K_FACTUAL` | `rag/retriever.py` | 8 |
| `K_COMPARATIVE` | `rag/retriever.py` | 10 |
| `RELEVANCE_THRESHOLD` | `rag/retriever.py` | 0.30 |
| `CHUNK_MAX_TOKENS` | `rag/indexer.py` | 400 |
| `MIN_CHUNK_CHARS` | `rag/indexer.py` | 20 |
| Embedding model | `rag/indexer.py` | `all-MiniLM-L6-v2` |

---

## Testing

No test suite exists yet. Manual verification only (see index verify command above).

When adding tests: use pytest, target 80% coverage, write tests first (TDD).

---

## Bot Restart Policy

After any change that affects output — prompts, speech patterns, retrieval logic, LLM parameters, response formatting, or event handling — **always restart the bot before considering the task done:**

```bash
pm2 restart raphael
```

Do not leave the old process running with stale code. If pm2 is not running, note it to the user.

---

## Automated Pipeline

Two Claude Code hooks run automatically during sessions — no manual steps needed:

| Trigger | What fires | Discord message |
|---------|-----------|-----------------|
| Linear issue marked **Done** | `PostToolUse: mcp__linear__save_issue` | One-liner task notice with Groq-generated summary |
| `git push` executed | `PostToolUse: Bash` | Changelog categorised by conventional commit prefix |

Both hooks call `scripts/discord_notify.py` and post to channel `1488606233807028275`.

**Hooks are session-bound** — they fire only during active Claude Code sessions. A push made outside Claude Code will not trigger a Discord message.

Hooks are configured in `.claude/settings.local.json` (machine-specific, not committed).

---

## Task Tracking

Tasks and ideas are tracked in **Linear** under the **Tensura** team, project **Raphael**.

**MANDATORY — no exceptions:**
- Every bug, idea, improvement, fix, or task that comes up in conversation gets a Linear issue immediately — before any code is written.
- When starting work on an issue, set its status to **In Progress**.
- When finishing, mark it **Done**.
- Do not batch issues up for later. Create them the moment they are identified.

**Milestone placement — think before assigning:**

Every issue must be placed in the correct milestone. Use the existing milestones as a guide:

| Milestone | What belongs here |
|-----------|------------------|
| Phase 0 — Setup Cleanup | Tooling, CI, config, dependency hygiene, infra housekeeping |
| Phase 1 — RAG Infrastructure | Core scraping, indexing, chunking, ChromaDB wiring |
| Phase 2 — Hardening | Reliability, rate limits, error handling, async safety |
| Phase 2 — Simple Recitation | Bot responds correctly to basic factual questions |
| Phase 2 — Answer Quality | Retrieval quality bugs: wrong chunks, missing data, bad scoring |
| Phase 3 — Advanced Reasoning | Enumeration queries, comparisons, multi-entity reasoning |
| Phase 4 — Chatbot Persona | Persona polish, fallback behaviour, non-wiki questions |

Rules:
- If the issue is a bugfix to an existing feature, place it in the same milestone as the original feature.
- If no existing milestone fits, create a new one first — don't leave issues unassigned or in the wrong phase.
- A feature that requires new retrieval logic (e.g. "list all X", "compare X vs Y") belongs in Phase 3, not Phase 2 — even if triggered by a Phase 2 bug report.

**Labels — always apply at least one:**

| Label | When to use |
|-------|------------|
| `Bug` | Something is broken or behaves incorrectly |
| `Feature` | New capability that doesn't exist yet |
| `Improvement` | Enhancement to existing behaviour (refactor, optimisation, UX) |
| `Bot` | Discord event handling, response formatting, user interaction |
| `LLM` | Prompts, Groq API, model routing, fallback logic |
| `RAG` | Scraper, indexer, retriever, ChromaDB |
| `Infrastructure` | PM2, cron, tooling, deployment, project setup |
| `Testing` | Test coverage, QA, verification |

Rules:
- Apply the *type* label (`Bug`, `Feature`, or `Improvement`) AND the *component* label (`Bot`, `LLM`, `RAG`, `Infrastructure`, `Testing`) where applicable.
- A single issue can and should carry multiple labels (e.g. a retrieval bug gets `Bug` + `RAG`).
