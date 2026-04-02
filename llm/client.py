import logging
import os
import re
import threading
import time

from groq import Groq, RateLimitError
from llm.prompts import RAPHAEL_SYSTEM_PROMPT, _sanitize_chunk, build_rag_prompt

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "qwen/qwen3-32b"
MAX_TOKENS = 1024
TEMPERATURE = 0.3

_client: Groq | None = None
logger = logging.getLogger(__name__)

_INSUFFICIENT_DATA = (
    "Insufficient data in my archives. "
    "Your query exceeds the bounds of available documentation. "
    "Recalibrate your question."
)

_API_OVERLOADED = (
    "My analytical processes are momentarily strained. "
    "Repeat your inquiry shortly — I shall resume calculations imminently."
)

# Circuit breaker: skip query expansion for this many seconds after a rate limit
_EXPANSION_COOLDOWN = 60.0
_rate_limit_lock = threading.Lock()
_last_rate_limit: float = 0.0


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        _client = Groq(api_key=api_key, max_retries=0)
    return _client


def _call(model: str, user_message: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RAPHAEL_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    if not response.choices:
        return _INSUFFICIENT_DATA
    content = response.choices[0].message.content or ""
    # Strip complete <think>…</think> blocks (e.g. qwen3-32b extended reasoning)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    # Strip truncated (unclosed) <think> blocks — happens when MAX_TOKENS is
    # exhausted mid-reasoning before the closing tag is produced
    content = re.sub(r"<think>.*", "", content, flags=re.DOTALL).strip()
    if not content:
        return _INSUFFICIENT_DATA
    logger.debug("LLM response [%s]: %s", model, content[:200])
    return content


def expand_query(question: str) -> list[str]:
    """Return 2–3 wiki-vocabulary rephrasing of the question for multi-shot retrieval.

    Used to bridge vocabulary gaps between player language ("tame mobs") and wiki
    terminology ("charm", "subjugate"). Returns an empty list on any failure so the
    caller can fall back to the original query without interruption.

    Skips the LLM call entirely when a recent rate limit was hit (circuit breaker)
    to conserve daily token budget.
    """
    with _rate_limit_lock:
        cooldown_active = time.monotonic() - _last_rate_limit < _EXPANSION_COOLDOWN
    if cooldown_active:
        logger.debug("Skipping query expansion — recent rate limit")
        return []
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Output only search queries, one per line. No commentary.",
                },
                {
                    "role": "user",
                    "content": (
                        "You are helping search a Tensura Minecraft mod wiki.\n"
                        f"User query: {_sanitize_chunk(question)}\n\n"
                        "Write 2-3 alternative phrasings of this query using wiki-style "
                        "terminology (stat names, skill/ability names, game mechanics). "
                        "One phrasing per line. No numbering, no explanation."
                    ),
                },
            ],
            max_tokens=80,
            temperature=0.3,
        )
        if not response.choices:
            return []
        content = response.choices[0].message.content or ""
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        content = re.sub(r"<think>.*", "", content, flags=re.DOTALL).strip()
        return [line.strip() for line in content.splitlines() if line.strip()][:3]
    except Exception:
        logger.warning("Query expansion failed — using original query only")
        return []


def answer(question: str, chunks: list[dict]) -> tuple[str, str]:
    """Generate a Raphael-persona answer grounded in the retrieved chunks.

    Tries the primary model first; falls back to the secondary on rate limit.
    Returns (response_text, model_used).
    """
    if not chunks:
        return _INSUFFICIENT_DATA, "none"

    user_message = build_rag_prompt(question, chunks)

    try:
        return _call(PRIMARY_MODEL, user_message), PRIMARY_MODEL
    except RateLimitError:
        with _rate_limit_lock:
            global _last_rate_limit
            _last_rate_limit = time.monotonic()
        logger.warning("Rate limit hit on %s, retrying with %s...", PRIMARY_MODEL, FALLBACK_MODEL)
        try:
            return _call(FALLBACK_MODEL, user_message), FALLBACK_MODEL
        except RateLimitError:
            logger.error("Fallback model %s also rate-limited", FALLBACK_MODEL)
            return _API_OVERLOADED, "rate_limited"
        except Exception:
            logger.exception("Fallback model %s failed (non-rate-limit)", FALLBACK_MODEL)
            return _API_OVERLOADED, "error"
