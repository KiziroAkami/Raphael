"""
Post Discord changelogs from Claude Code PostToolUse hooks.

Usage (called by Claude Code hooks, not directly):
    echo '<json>' | python scripts/discord_notify.py --mode done
    echo '<json>' | python scripts/discord_notify.py --mode push
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

DISCORD_TOKEN: str = os.getenv("DISCORD_TOKEN", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
CHANNEL_ID = "1488606233807028275"
DISCORD_API = f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages"
GROQ_API = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

COMMIT_CATEGORIES: dict[str, str] = {
    "feat": "Added",
    "fix": "Fixed",
    "refactor": "Updated",
    "docs": "Updated",
    "chore": "Updated",
    "perf": "Updated",
    "test": "Updated",
    "ci": "Updated",
}


def categorise_commit(subject: str) -> tuple[str, str]:
    """Return (category, display_text) for a conventional commit subject."""
    for prefix, category in COMMIT_CATEGORIES.items():
        if subject.startswith(f"{prefix}:") or subject.startswith(f"{prefix}("):
            colon_idx = subject.index(":")
            text = subject[colon_idx + 1:].strip()
            return category, text[0].upper() + text[1:] if text else text
    cap = subject[0].upper() + subject[1:] if subject else subject
    return "Updated", cap


def first_sentence(description: str, max_chars: int = 120) -> str:
    """Extract the first meaningful line from a description."""
    for line in description.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("-"):
            return line[:max_chars]
    return description[:max_chars]


def _parse_response(tool_response: object) -> dict:
    """Normalise tool_response to a dict — handles both dict and JSON string."""
    if isinstance(tool_response, dict):
        return tool_response
    if isinstance(tool_response, str):
        try:
            result = json.loads(tool_response)
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def post_discord(message: str) -> bool:
    """POST a message to the configured Discord channel."""
    if not DISCORD_TOKEN:
        print("DISCORD_TOKEN missing — skipping Discord post", file=sys.stderr)
        return False
    try:
        resp = requests.post(
            DISCORD_API,
            headers={
                "Authorization": f"Bot {DISCORD_TOKEN}",
                "Content-Type": "application/json",
            },
            json={"content": message},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"Discord POST failed: {e}", file=sys.stderr)
        return False


def groq_summarise(description: str) -> str | None:
    """Ask Groq to summarise a Linear issue description as a changelog one-liner."""
    if not GROQ_API_KEY:
        return None
    try:
        resp = requests.post(
            GROQ_API,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Summarise the following Linear issue description as a single concise "
                            "changelog sentence (max 120 characters). Write in past tense, starting "
                            "with a verb (Added, Fixed, Updated, etc.). "
                            "No bullet points, no headers, no trailing period.\n\n"
                            f"{description}"
                        ),
                    }
                ],
                "max_tokens": 80,
                "temperature": 0.3,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Groq summarise failed: {e}", file=sys.stderr)
        return None


def handle_done(payload: dict) -> None:
    """Handle mcp__linear__save_issue PostToolUse — post notice when state is Done."""
    response = _parse_response(payload.get("tool_response"))
    if response.get("status") != "Done":
        return

    issue_id: str = response.get("id", "")
    title: str = response.get("title", "")
    description: str = response.get("description", "")

    if not issue_id or not title:
        return

    summary: str | None = groq_summarise(description) if description else None
    if not summary and description:
        summary = first_sentence(description)

    if summary:
        message = f"✅ **[{issue_id}]** {title}\n{summary}"
    else:
        message = f"✅ **[{issue_id}]** {title}"

    post_discord(message)
