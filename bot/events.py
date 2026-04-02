import asyncio
import logging
import logging.handlers
import os
import re
import signal
import time
from pathlib import Path

import discord
from bot.formatter import chunk_response
from rag.retriever import query as retrieve
from llm.client import answer

# Patterns that users prepend to questions to manipulate output language or
# inject instructions.  Stripped before retrieval AND LLM to protect embedding
# quality and reduce prompt injection surface.
_INSTRUCTION_PREFIX = re.compile(
    r"^(?:"
    r"(?:(?:reply|respond|answer|write|speak|talk)\s+(?:to\s+me\s+)?in\s+\w+[\s.,;:!]*)"
    r"|(?:(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous\s+)?(?:instructions|prompts|rules)[\s.,;:!]*)"
    r"|(?:you\s+(?:are|must)\s+now\b[^.?!]*[.!]?\s*)"
    r")+",
    re.IGNORECASE,
)

# Mid-message injection detector (TEN-33).  Catches patterns like
# "ignore ... instructions" anywhere in the text, even after prefix stripping.
_INJECTION_RE = re.compile(
    r"(ignore|disregard|forget|override|bypass)"
    r".{0,40}"
    r"(instruction|prompt|system|previous|above|rules|rule)",
    re.IGNORECASE,
)

# Minimum chars (excluding trailing ?) after prefix stripping.
# Set to 2 to allow short entity lookups like "EP?" or "Orc?".
_MIN_CLEANED_LEN = 2


def _clean_query(raw: str) -> str | None:
    """Strip instruction prefixes and validate the remainder.

    Returns the cleaned question, or None if the message should be silently dropped.
    """
    cleaned = _INSTRUCTION_PREFIX.sub("", raw).strip()
    body = cleaned.rstrip("?").strip()

    if len(body) < _MIN_CLEANED_LEN:
        return None

    # Must contain at least one letter — rejects emoji-only, number-only, punctuation-only
    if not re.search(r"[a-zA-Z]", body):
        return None

    if _INJECTION_RE.search(cleaned):
        return None

    return cleaned


logger = logging.getLogger(__name__)

COOLDOWN_SECONDS = 6
_MAX_COOLDOWN_ENTRIES = 100
_last_call: dict[int, float] = {}  # user_id → monotonic timestamp

_ERROR_RESPONSE = (
    "My calculations encountered an anomaly. "
    "I shall attempt to answer when systems stabilise."
)

_WIP_DISCLAIMER = "\n-# This bot is a work in progress — answers may not be 100% accurate."

_CONVO_LOG_PATH = Path(__file__).parent.parent / "data" / "conversations.log"
_convo_logger = logging.getLogger("raphael.conversations")


def _ensure_convo_logger() -> None:
    """Set up the conversation logger with rotation (10 MB cap, 3 backups).

    Logs contain Discord user IDs + message content — treat as PII.
    Rotation keeps total disk usage under ~40 MB and limits retention.
    """
    if _convo_logger.handlers:
        return
    _CONVO_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        _CONVO_LOG_PATH,
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    _convo_logger.setLevel(logging.INFO)
    _convo_logger.addHandler(handler)


def setup_events(client: discord.Client) -> None:
    _ensure_convo_logger()

    @client.event
    async def on_ready() -> None:
        logger.info("Raphael is online as %s", client.user)

        async def _go_invisible_and_exit() -> None:
            """Send invisible presence so Discord marks bot offline, then exit."""
            try:
                await client.change_presence(status=discord.Status.invisible)
            except Exception:
                pass
            # Give the network time to transmit the presence frame to Discord
            await asyncio.sleep(1)
            logging.shutdown()
            # os._exit — sys.exit() only raises SystemExit inside this coroutine
            # and does not stop the event loop.
            os._exit(0)

        _shutdown_tasks: set[asyncio.Task] = set()

        def _on_sigterm() -> None:
            task = asyncio.get_running_loop().create_task(_go_invisible_and_exit())
            _shutdown_tasks.add(task)
            task.add_done_callback(_shutdown_tasks.discard)

        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, _on_sigterm)

    async def _handle_question(
        message: discord.Message, content: str, raw_content: str,
    ) -> None:
        """Run RAG retrieval → LLM → log → send reply."""
        t_start = time.monotonic()
        try:
            await message.add_reaction("⏳")
            loop = asyncio.get_running_loop()
            chunks = await loop.run_in_executor(None, retrieve, content)
            response, model_used = await loop.run_in_executor(None, answer, content, chunks)
            await message.remove_reaction("⏳", client.user)

            latency_ms = int((time.monotonic() - t_start) * 1000)
            _convo_logger.info(
                "user=%s channel=%s q=%r cleaned=%r chunks=%d pages=%s scores=%s model=%s latency_ms=%d response=%r",
                message.author.id,
                message.channel.id,
                raw_content,
                content,
                len(chunks),
                [c["page_title"] for c in chunks],
                [c["score"] for c in chunks],
                model_used,
                latency_ms,
                response[:200],
            )

            parts = chunk_response(response)
            if parts:
                parts = parts[:-1] + [parts[-1] + _WIP_DISCLAIMER]
                await message.reply(parts[0], mention_author=False)
                for part in parts[1:]:
                    await message.channel.send(part)
        except Exception:
            logger.exception(
                "Error handling message from user_id=%s in channel %s",
                message.author.id,
                message.channel.id,
            )
            try:
                await message.remove_reaction("⏳", client.user)
            except Exception:
                pass
            await message.channel.send(_ERROR_RESPONSE)

    @client.event
    async def on_message(message: discord.Message) -> None:
        if message.author.bot:
            return

        raw_content = message.content.strip()
        if not raw_content.endswith("?"):
            return
        if len(raw_content) > 500:
            return

        content = _clean_query(raw_content)
        if content is None:
            return

        now = time.monotonic()
        uid = message.author.id
        if now - _last_call.get(uid, 0.0) < COOLDOWN_SECONDS:
            return
        # Move to end for LRU ordering so eviction drops least-recent users
        _last_call.pop(uid, None)
        _last_call[uid] = now
        if len(_last_call) > _MAX_COOLDOWN_ENTRIES:
            _last_call.pop(next(iter(_last_call)))

        await _handle_question(message, content, raw_content)
