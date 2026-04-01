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
import time
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
    """Normalise tool_response to a dict.

    Claude Code hook payloads wrap MCP responses in a content list:
      [{"type": "text", "text": "<JSON string>"}]
    This function unwraps that before falling through to dict/string handling.
    """
    if isinstance(tool_response, list):
        for item in tool_response:
            if isinstance(item, dict) and item.get("type") == "text":
                tool_response = item.get("text", "")
                break
        else:
            return {}
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
    """POST a message to the configured Discord channel.

    Retries once on HTTP 429 using the retry_after value from Discord's response body.
    """
    if not DISCORD_TOKEN:
        print("DISCORD_TOKEN missing — skipping Discord post", file=sys.stderr)
        return False
    headers = {
        "Authorization": f"Bot {DISCORD_TOKEN}",
        "Content-Type": "application/json",
    }
    for attempt in range(2):
        try:
            resp = requests.post(
                DISCORD_API,
                headers=headers,
                json={"content": message},
                timeout=10,
            )
            if resp.status_code == 429:
                wait = float(resp.json().get("retry_after", 1))
                print(f"Discord rate limited — retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return True
        except Exception as e:
            print(f"Discord POST failed: {e}", file=sys.stderr)
            return False
    print("Discord POST failed after retry", file=sys.stderr)
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


def get_commits_since_origin(branch: str) -> list[str]:
    """Return commit subjects between origin/branch and HEAD."""
    try:
        result = subprocess.run(
            ["git", "log", f"origin/{branch}..{branch}", "--pretty=format:%s"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception as e:
        print(f"git log failed: {e}", file=sys.stderr)
        return []


def extract_branch(command: str) -> str | None:
    """Extract branch name from a git push command, falling back to current branch."""
    parts = command.split()
    # Collect positional args after "git push", skipping flags and refspecs.
    # Skip flags and refspecs (e.g. :branch = delete, local:remote = explicit mapping).
    # HEAD:refs/heads/branch form used in some CI pipelines will not match — acceptable trade-off.
    positional = [p for p in parts[2:] if not p.startswith("-") and ":" not in p]
    # positional[0] = remote, positional[1] = branch (if explicitly given)
    if len(positional) >= 2:
        return positional[1]
    # Fall back to current branch
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def handle_push(payload: dict) -> None:
    """Handle Bash PostToolUse — post changelog when command is a git push."""
    tool_input = payload.get("tool_input", {})
    command: str = tool_input.get("command", "") if isinstance(tool_input, dict) else ""

    parts = command.split()
    if not (len(parts) >= 2 and parts[0] == "git" and parts[1] == "push"):
        return
    if "--dry-run" in parts:
        return
    if any(p.startswith(":") for p in parts) or "--delete" in parts or "-d" in parts:
        return

    branch = extract_branch(command)
    if not branch:
        return

    commits = get_commits_since_origin(branch)
    if not commits:
        return

    buckets: dict[str, list[str]] = {"Added": [], "Updated": [], "Fixed": []}
    for subject in commits:
        category, text = categorise_commit(subject)
        buckets[category].append(text)

    lines = [f"📦 **Changelog** · `{branch}`"]
    for section in ("Added", "Updated", "Fixed"):
        if buckets[section]:
            lines.append(f"\n**{section}**")
            for item in buckets[section]:
                lines.append(f"- {item}")

    post_discord("\n".join(lines))


def handle_done(payload: dict) -> None:
    """Handle mcp__linear__save_issue PostToolUse — post notice when state is Done."""
    response = _parse_response(payload.get("tool_response"))
    if response.get("status") != "Done":
        return

    issue_id: str = response.get("id", "")
    title: str = response.get("title", "")
    description: str = response.get("description", "")
    milestone: str = (response.get("projectMilestone") or {}).get("name", "")

    if not issue_id or not title:
        return

    summary: str | None = groq_summarise(description) if description else None
    if not summary and description:
        summary = first_sentence(description)

    header = f"✅ **[{issue_id}]** {title}"
    if milestone:
        header += f" · `{milestone}`"

    message = f"{header}\n{summary}" if summary else header
    post_discord(message)


def main() -> None:
    if "--mode" not in sys.argv:
        print("Usage: discord_notify.py --mode done|push", file=sys.stderr)
        sys.exit(0)

    mode_idx = sys.argv.index("--mode")
    if mode_idx + 1 >= len(sys.argv):
        print("--mode requires an argument", file=sys.stderr)
        sys.exit(0)

    mode = sys.argv[mode_idx + 1]

    try:
        payload = json.load(sys.stdin)
    except Exception as e:
        print(f"Failed to parse hook payload: {e}", file=sys.stderr)
        sys.exit(0)

    if mode == "done":
        handle_done(payload)
    elif mode == "push":
        handle_push(payload)
    else:
        print(f"Unknown mode: {mode!r}", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
