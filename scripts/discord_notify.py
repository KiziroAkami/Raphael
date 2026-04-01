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
