import logging
import os
import re
import threading
import time

from groq import APIStatusError, Groq, RateLimitError
from llm.prompts import RAPHAEL_SYSTEM_PROMPT, _sanitize_chunk, build_rag_prompt

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "qwen/qwen3-32b"
TERTIARY_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
LAST_RESORT_MODEL = "llama-3.1-8b-instant"
MODEL_CHAIN: list[str] = [PRIMARY_MODEL, FALLBACK_MODEL, TERTIARY_MODEL, LAST_RESORT_MODEL]
MAX_TOKENS = 1024
TEMPERATURE = 0.1

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
    # Strip markdown headers (### Heading) — some models (llama-4-scout) emit them
    # despite the system prompt saying "prefer flowing analytical prose"
    content = re.sub(r"^#{1,4}\s+", "", content, flags=re.MULTILINE)
    # Strip horizontal rules (---) that some models insert
    content = re.sub(r"^-{3,}\s*$", "", content, flags=re.MULTILINE)
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
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
            temperature=0.3,  # intentionally higher than TEMPERATURE — diversity helps expansion
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


def answer(
    question: str,
    chunks: list[dict],
    history: list[tuple[str, str]] | None = None,
) -> tuple[str, str]:
    """Generate a Raphael-persona answer grounded in the retrieved chunks.

    Walks MODEL_CHAIN in order, falling back on RateLimitError.
    Returns (response_text, model_used).

    With empty chunks, still calls the LLM so identity/meta questions can be
    answered from system-prompt background knowledge. (TEN-179.) The prompt
    builder injects an explicit "no wiki context" instruction in that case.
    """
    user_message = build_rag_prompt(question, chunks, history=history)

    for i, model in enumerate(MODEL_CHAIN):
        try:
            return _call(model, user_message), model
        except RateLimitError:
            with _rate_limit_lock:
                global _last_rate_limit
                _last_rate_limit = time.monotonic()
            next_model = MODEL_CHAIN[i + 1] if i + 1 < len(MODEL_CHAIN) else None
            if next_model:
                logger.warning("Rate limit hit on %s, retrying with %s...", model, next_model)
            else:
                logger.error("All %d models rate-limited", len(MODEL_CHAIN))
                return _API_OVERLOADED, "rate_limited"
        except APIStatusError as e:
            # 413 Payload Too Large — request exceeds model's TPM limit.
            # Try next model which may have a higher TPM allowance.
            if e.status_code == 413:
                next_model = MODEL_CHAIN[i + 1] if i + 1 < len(MODEL_CHAIN) else None
                if next_model:
                    logger.warning("Payload too large for %s (HTTP 413), trying %s...", model, next_model)
                else:
                    logger.error("Payload too large for all %d models", len(MODEL_CHAIN))
                    return _API_OVERLOADED, "payload_too_large"
            else:
                logger.exception("Model %s failed (status %d)", model, e.status_code)
                return _API_OVERLOADED, "error"
        except Exception:
            logger.exception("Model %s failed (non-API)", model)
            return _API_OVERLOADED, "error"

    return _API_OVERLOADED, "rate_limited"
