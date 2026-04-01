import logging
import os
import re
import time

from groq import Groq, RateLimitError
from llm.prompts import RAPHAEL_SYSTEM_PROMPT, build_rag_prompt

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "qwen/qwen3-32b"
MAX_TOKENS = 1024
TEMPERATURE = 0.3

_client: Groq | None = None
logger = logging.getLogger(__name__)

_INSUFFICIENT_DATA = (
    "Insufficient data in Raphael's archives. "
    "This query exceeds the bounds of available documentation. "
    "Recalibrate your question."
)


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        _client = Groq(api_key=api_key)
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
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if not content:
        return _INSUFFICIENT_DATA
    logger.debug("LLM response [%s]: %s", model, content[:200])
    return content


def expand_query(question: str) -> list[str]:
    """Return 2–3 wiki-vocabulary rephrasing of the question for multi-shot retrieval.

    Used to bridge vocabulary gaps between player language ("tame mobs") and wiki
    terminology ("charm", "subjugate"). Returns an empty list on any failure so the
    caller can fall back to the original query without interruption.
    """
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are helping search a Tensura Minecraft mod wiki.\n"
                        f"User query: {question}\n\n"
                        "Write 2-3 alternative phrasings of this query using wiki-style "
                        "terminology (stat names, skill/ability names, game mechanics). "
                        "One phrasing per line. No numbering, no explanation."
                    ),
                }
            ],
            max_tokens=80,
            temperature=0.3,
        )
        if not response.choices:
            return []
        content = response.choices[0].message.content or ""
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return [line.strip() for line in content.splitlines() if line.strip()][:3]
    except Exception:
        logger.warning("Query expansion failed — using original query only")
        return []


def answer(question: str, chunks: list[dict]) -> str:
    """Generate a Raphael-persona answer grounded in the retrieved chunks.

    Tries the primary model first; falls back to the secondary on rate limit.
    """
    if not chunks:
        return _INSUFFICIENT_DATA

    user_message = build_rag_prompt(question, chunks)

    try:
        return _call(PRIMARY_MODEL, user_message)
    except RateLimitError:
        logger.warning("Rate limit hit on %s, retrying with %s...", PRIMARY_MODEL, FALLBACK_MODEL)
        time.sleep(1)
        try:
            return _call(FALLBACK_MODEL, user_message)
        except Exception:
            logger.exception("Fallback model %s also failed", FALLBACK_MODEL)
            return _INSUFFICIENT_DATA
