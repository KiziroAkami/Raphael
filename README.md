# Raphael — Tensura Minecraft Mod Discord Bot

A Discord bot themed as Raphael, Lord of Wisdom. Answers questions about the [Tensura Minecraft mod](https://tensura.wiki.gg/) by retrieving data from the wiki and responding in Raphael's voice. Any message ending with `?` triggers a response.

---

## First-Time Setup

**Prerequisites:** Python 3.11+, a Discord bot token, a Groq API key.

**1. Clone and enter the project**
```bash
cd ~/Projects/Raphael
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
You'll see `(.venv)` in your prompt. Run this activation command every time you open a new terminal.

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure secrets**
```bash
cp .env.example .env
```
Open `.env` and fill in your `DISCORD_TOKEN` and `GROQ_API_KEY`.

**5. Build the wiki index** *(takes ~7 minutes — fetches all ~836 wiki pages)*
```bash
python scripts/build_index.py
```

**6. Verify the index built correctly**
```bash
python -c "from rag.indexer import get_collection; c = get_collection(); print(len(c.get()['ids']), 'chunks indexed')"
```
Expect 2,500+ chunks.

**7. Install pm2** *(keeps the bot running persistently — survives terminal close and reboots)*
```bash
npm install -g pm2
```

**8. Start the bot via pm2**
```bash
pm2 start ecosystem.config.cjs
pm2 save
pm2 startup   # follow the printed command to enable auto-start on reboot
```

This starts three processes:
- **raphael** — the Discord bot
- **raphael-caffeinate** — prevents macOS idle/display sleep (`caffeinate -di`)
- **raphael-sync** — a cron job that runs `--incremental` wiki sync daily at 03:00

> **Sleep note:** `caffeinate -di` prevents idle and display sleep while the bot is running.
> If the lid is closed on battery, macOS may still sleep. For guaranteed 24/7 uptime,
> consider running the bot on a VPS instead.

---

## Daily Usage

**Bot control:**
```bash
pm2 status                  # check both raphael and raphael-sync
pm2 logs raphael            # tail live bot logs
pm2 logs raphael-sync       # tail sync job logs
pm2 restart raphael         # restart after code changes
pm2 stop raphael            # stop the bot
```

**Index management:**

| Command | When to use |
|---------|-------------|
| `python scripts/build_index.py --incremental` | New or edited wiki pages since last run (runs automatically at 03:00 — use this to trigger manually outside the cron) |
| `python scripts/build_index.py --cached` | Re-index from the local disk cache without hitting the wiki API (useful after code changes to the chunker/indexer) |
| `python scripts/build_index.py --refresh` | Force re-fetch every page from the wiki and rebuild the index from scratch |
| `python scripts/build_index.py` | First-time full build only — fetches all pages, skipping any already on disk |

> **Note:** `--incremental` picks up both new pages and edits via the MediaWiki `recentchanges` API. It is the right command whenever you want to pull in anything that changed on the wiki since the last sync.
