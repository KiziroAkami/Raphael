import logging
import os
import re
import time

from groq import Groq, RateLimitError
from llm.prompts import RAPHAEL_SYSTEM_PROMPT, build_rag_prompt

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "qwen/qwen3-32b"
MAX_TOKENS = 1024
TEMPERATURE = 0.7

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
    return content if content else _INSUFFICIENT_DATA


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
