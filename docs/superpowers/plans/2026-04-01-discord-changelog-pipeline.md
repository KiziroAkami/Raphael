# Discord Changelog Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two Claude Code PostToolUse hooks that automatically post Discord messages when a Linear issue is marked Done (brief task notice) or when a `git push` runs (categorised changelog).

**Architecture:** A single script `scripts/discord_notify.py` with `--mode done|push` reads the hook JSON payload from stdin, decides whether to act, formats a message, and POSTs to the Discord REST API using the existing `DISCORD_TOKEN`. Two hooks in `.claude/settings.local.json` wire the script to the right tool events.

**Tech Stack:** Python 3.14, `requests`, `python-dotenv`, Discord REST API v10, Groq API (`llama-3.3-70b-versatile`), Claude Code PostToolUse hooks.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `scripts/discord_notify.py` | Main script — both modes, Discord POST, Groq summarise |
| Create | `tests/scripts/test_discord_notify.py` | Full unit test suite |
| Create | `tests/scripts/__init__.py` | Package marker |
| Create | `tests/__init__.py` | Package marker |
| Create | `.claude/settings.local.json` | PostToolUse hook wiring (machine-specific, not committed) |
| Modify | `.gitignore` | Exclude `.claude/settings.local.json` |
| Modify | `CLAUDE.md` | Document the automated pipeline |

---

## Task 1: Bootstrap test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/scripts/__init__.py`
- Create: `tests/scripts/test_discord_notify.py`

- [ ] **Step 1: Create package markers**

```bash
mkdir -p tests/scripts
touch tests/__init__.py tests/scripts/__init__.py
```

- [ ] **Step 2: Create the test file with imports and fixtures**

Create `tests/scripts/test_discord_notify.py`:

```python
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# Make sure the project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

- [ ] **Step 3: Verify pytest can collect the file**

```bash
.venv/bin/python -m pytest tests/scripts/test_discord_notify.py --collect-only
```

Expected output: `0 tests collected` (no errors).

- [ ] **Step 4: Commit the scaffold**

```bash
git add tests/
git commit -m "test: scaffold test directory for discord_notify"
```

---

## Task 2: TDD — `categorise_commit`

**Files:**
- Modify: `tests/scripts/test_discord_notify.py`
- Create: `scripts/discord_notify.py` (stub only)

- [ ] **Step 1: Write failing tests for `categorise_commit`**

Append to `tests/scripts/test_discord_notify.py`:

```python
from scripts.discord_notify import categorise_commit


class TestCategoriseCommit:
    def test_feat_prefix_returns_added(self):
        category, text = categorise_commit("feat: add wiki scraper")
        assert category == "Added"
        assert text == "Add wiki scraper"

    def test_fix_prefix_returns_fixed(self):
        category, text = categorise_commit("fix: handle rate limit backoff")
        assert category == "Fixed"
        assert text == "Handle rate limit backoff"

    def test_docs_prefix_returns_updated(self):
        category, text = categorise_commit("docs: clarify index management")
        assert category == "Updated"
        assert text == "Clarify index management"

    def test_chore_prefix_returns_updated(self):
        category, text = categorise_commit("chore: add pm2 config")
        assert category == "Updated"
        assert text == "Add pm2 config"

    def test_refactor_prefix_returns_updated(self):
        category, text = categorise_commit("refactor: split scraper module")
        assert category == "Updated"
        assert text == "Split scraper module"

    def test_scoped_feat_strips_scope(self):
        category, text = categorise_commit("feat(rag): add embeddings")
        assert category == "Added"
        assert text == "Add embeddings"

    def test_unknown_prefix_falls_back_to_updated(self):
        category, text = categorise_commit("wip: half done thing")
        assert category == "Updated"
        assert text == "Wip: half done thing"

    def test_no_prefix_falls_back_to_updated(self):
        category, text = categorise_commit("random commit message")
        assert category == "Updated"
        assert text == "Random commit message"
```

- [ ] **Step 2: Create minimal script stub so import doesn't fail**

Create `scripts/discord_notify.py`:

```python
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
    pass  # implement in Task 2 Step 4
```

- [ ] **Step 3: Run tests and confirm they fail**

```bash
.venv/bin/python -m pytest tests/scripts/test_discord_notify.py::TestCategoriseCommit -v
```

Expected: all tests FAIL (TypeError or assertion error from `pass`).

- [ ] **Step 4: Implement `categorise_commit`**

Replace the `categorise_commit` stub in `scripts/discord_notify.py`:

```python
def categorise_commit(subject: str) -> tuple[str, str]:
    """Return (category, display_text) for a conventional commit subject."""
    for prefix, category in COMMIT_CATEGORIES.items():
        if subject.startswith(f"{prefix}:") or subject.startswith(f"{prefix}("):
            colon_idx = subject.index(":")
            text = subject[colon_idx + 1:].strip()
            return category, text[0].upper() + text[1:] if text else text
    cap = subject[0].upper() + subject[1:] if subject else subject
    return "Updated", cap
```

- [ ] **Step 5: Run tests and confirm they pass**

```bash
.venv/bin/python -m pytest tests/scripts/test_discord_notify.py::TestCategoriseCommit -v
```

Expected: 8 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/discord_notify.py tests/scripts/test_discord_notify.py tests/__init__.py tests/scripts/__init__.py
git commit -m "feat: add categorise_commit with tests"
```

---

## Task 3: TDD — `first_sentence` and `_parse_response`

**Files:**
- Modify: `tests/scripts/test_discord_notify.py`
- Modify: `scripts/discord_notify.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/scripts/test_discord_notify.py`:

```python
from scripts.discord_notify import first_sentence, _parse_response


class TestFirstSentence:
    def test_returns_first_non_empty_line(self):
        desc = "Added a Discord webhook.\n\nMore details."
        assert first_sentence(desc) == "Added a Discord webhook."

    def test_skips_markdown_headers(self):
        desc = "## Goal\nAdded PostToolUse hooks."
        assert first_sentence(desc) == "Added PostToolUse hooks."

    def test_skips_bullet_lines(self):
        desc = "- some bullet\nFixed the scraper."
        assert first_sentence(desc) == "Fixed the scraper."

    def test_truncates_to_max_chars(self):
        desc = "A" * 200
        assert len(first_sentence(desc)) == 120

    def test_empty_description_returns_empty(self):
        assert first_sentence("") == ""


class TestParseResponse:
    def test_returns_dict_unchanged(self):
        data = {"id": "TEN-1", "status": "Done"}
        assert _parse_response(data) == data

    def test_parses_json_string(self):
        data = '{"id": "TEN-1", "status": "Done"}'
        assert _parse_response(data) == {"id": "TEN-1", "status": "Done"}

    def test_returns_empty_dict_on_invalid_json_string(self):
        assert _parse_response("not json") == {}

    def test_returns_empty_dict_on_none(self):
        assert _parse_response(None) == {}
```

- [ ] **Step 2: Run and confirm tests fail**

```bash
.venv/bin/python -m pytest tests/scripts/test_discord_notify.py::TestFirstSentence tests/scripts/test_discord_notify.py::TestParseResponse -v
```

Expected: ImportError or AttributeError — functions not yet defined.

- [ ] **Step 3: Implement `first_sentence` and `_parse_response`**

Append to `scripts/discord_notify.py` (after `categorise_commit`):

```python
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
```

- [ ] **Step 4: Run and confirm tests pass**

```bash
.venv/bin/python -m pytest tests/scripts/test_discord_notify.py::TestFirstSentence tests/scripts/test_discord_notify.py::TestParseResponse -v
```

Expected: 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/discord_notify.py tests/scripts/test_discord_notify.py
git commit -m "feat: add first_sentence and _parse_response helpers with tests"
```

---

## Task 4: TDD — `handle_done`

**Files:**
- Modify: `tests/scripts/test_discord_notify.py`
- Modify: `scripts/discord_notify.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/scripts/test_discord_notify.py`:

```python
from scripts.discord_notify import handle_done


class TestHandleDone:
    def _payload(self, status: str = "Done", issue_id: str = "TEN-41",
                 title: str = "Add Discord hooks", description: str = "Some description.") -> dict:
        return {
            "tool_name": "mcp__linear__save_issue",
            "tool_input": {"id": issue_id, "state": status},
            "tool_response": {
                "id": issue_id,
                "title": title,
                "status": status,
                "description": description,
            },
        }

    @patch("scripts.discord_notify.post_discord")
    @patch("scripts.discord_notify.groq_summarise", return_value="Added PostToolUse hooks for Discord.")
    def test_posts_message_when_done(self, mock_groq, mock_post):
        handle_done(self._payload())
        mock_post.assert_called_once()
        msg = mock_post.call_args[0][0]
        assert "TEN-41" in msg
        assert "Add Discord hooks" in msg
        assert "Added PostToolUse hooks for Discord." in msg

    @patch("scripts.discord_notify.post_discord")
    def test_skips_when_not_done(self, mock_post):
        handle_done(self._payload(status="In Progress"))
        mock_post.assert_not_called()

    @patch("scripts.discord_notify.post_discord")
    @patch("scripts.discord_notify.groq_summarise", return_value=None)
    def test_falls_back_to_first_sentence_when_groq_fails(self, mock_groq, mock_post):
        handle_done(self._payload(description="Fixed the scraper rate limit."))
        mock_post.assert_called_once()
        msg = mock_post.call_args[0][0]
        assert "Fixed the scraper rate limit." in msg

    @patch("scripts.discord_notify.post_discord")
    @patch("scripts.discord_notify.groq_summarise", return_value=None)
    def test_posts_title_only_when_no_description(self, mock_groq, mock_post):
        handle_done(self._payload(description=""))
        mock_post.assert_called_once()
        msg = mock_post.call_args[0][0]
        assert "TEN-41" in msg

    @patch("scripts.discord_notify.post_discord")
    def test_skips_when_missing_id(self, mock_post):
        payload = self._payload()
        payload["tool_response"]["id"] = ""
        handle_done(payload)
        mock_post.assert_not_called()
```

- [ ] **Step 2: Run and confirm tests fail**

```bash
.venv/bin/python -m pytest tests/scripts/test_discord_notify.py::TestHandleDone -v
```

Expected: ImportError — `handle_done` not yet defined.

- [ ] **Step 3: Implement `handle_done` and leaf functions**

Append to `scripts/discord_notify.py`:

```python
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
```

- [ ] **Step 4: Run and confirm tests pass**

```bash
.venv/bin/python -m pytest tests/scripts/test_discord_notify.py::TestHandleDone -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/discord_notify.py tests/scripts/test_discord_notify.py
git commit -m "feat: implement handle_done with Groq summary and Discord post"
```

---

## Task 5: TDD — `handle_push`

**Files:**
- Modify: `tests/scripts/test_discord_notify.py`
- Modify: `scripts/discord_notify.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/scripts/test_discord_notify.py`:

```python
from scripts.discord_notify import handle_push


class TestHandlePush:
    def _payload(self, command: str) -> dict:
        return {
            "tool_name": "Bash",
            "tool_input": {"command": command},
            "tool_response": {"output": ""},
        }

    @patch("scripts.discord_notify.post_discord")
    @patch("scripts.discord_notify.get_commits_since_origin", return_value=["feat: add scraper", "fix: handle 429"])
    @patch("scripts.discord_notify.extract_branch", return_value="dev")
    def test_posts_changelog_on_push(self, mock_branch, mock_commits, mock_post):
        handle_push(self._payload("git push origin dev"))
        mock_post.assert_called_once()
        msg = mock_post.call_args[0][0]
        assert "**Added**" in msg
        assert "Add scraper" in msg
        assert "**Fixed**" in msg
        assert "Handle 429" in msg

    @patch("scripts.discord_notify.post_discord")
    def test_skips_non_push_command(self, mock_post):
        handle_push(self._payload("git status"))
        mock_post.assert_not_called()

    @patch("scripts.discord_notify.post_discord")
    def test_skips_dry_run(self, mock_post):
        handle_push(self._payload("git push origin dev --dry-run"))
        mock_post.assert_not_called()

    @patch("scripts.discord_notify.post_discord")
    def test_skips_branch_delete_colon(self, mock_post):
        handle_push(self._payload("git push origin :dev"))
        mock_post.assert_not_called()

    @patch("scripts.discord_notify.post_discord")
    def test_skips_branch_delete_flag(self, mock_post):
        handle_push(self._payload("git push origin --delete dev"))
        mock_post.assert_not_called()

    @patch("scripts.discord_notify.post_discord")
    @patch("scripts.discord_notify.get_commits_since_origin", return_value=[])
    @patch("scripts.discord_notify.extract_branch", return_value="dev")
    def test_skips_when_no_commits(self, mock_branch, mock_commits, mock_post):
        handle_push(self._payload("git push origin dev"))
        mock_post.assert_not_called()

    @patch("scripts.discord_notify.post_discord")
    @patch("scripts.discord_notify.get_commits_since_origin", return_value=["docs: update readme"])
    @patch("scripts.discord_notify.extract_branch", return_value="dev")
    def test_omits_empty_sections(self, mock_branch, mock_commits, mock_post):
        handle_push(self._payload("git push origin dev"))
        mock_post.assert_called_once()
        msg = mock_post.call_args[0][0]
        assert "**Added**" not in msg
        assert "**Fixed**" not in msg
        assert "**Updated**" in msg
        assert "Update readme" in msg
```

- [ ] **Step 2: Run and confirm tests fail**

```bash
.venv/bin/python -m pytest tests/scripts/test_discord_notify.py::TestHandlePush -v
```

Expected: ImportError — `handle_push`, `get_commits_since_origin`, `extract_branch` not yet defined.

- [ ] **Step 3: Implement `handle_push` and helpers**

Append to `scripts/discord_notify.py`:

```python
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
    # Try to parse positional branch arg: git push <remote> <branch>
    if len(parts) >= 4:
        for part in parts[3:]:
            if not part.startswith("-") and ":" not in part:
                return part
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
```

- [ ] **Step 4: Run and confirm tests pass**

```bash
.venv/bin/python -m pytest tests/scripts/test_discord_notify.py::TestHandlePush -v
```

Expected: 8 tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/discord_notify.py tests/scripts/test_discord_notify.py
git commit -m "feat: implement handle_push with commit categorisation"
```

---

## Task 6: Wire up `main` entry point

**Files:**
- Modify: `scripts/discord_notify.py`

- [ ] **Step 1: Append `main` to `scripts/discord_notify.py`**

```python
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
```

- [ ] **Step 2: Smoke-test done mode with a fake payload**

```bash
echo '{"tool_response": {"id": "TEN-99", "title": "Test issue", "status": "In Progress", "description": ""}}' \
  | .venv/bin/python scripts/discord_notify.py --mode done
```

Expected: no output, no crash (state is not Done — silent skip).

- [ ] **Step 3: Smoke-test push mode with a non-push command**

```bash
echo '{"tool_input": {"command": "git status"}}' \
  | .venv/bin/python scripts/discord_notify.py --mode push
```

Expected: no output, no crash.

- [ ] **Step 4: Commit**

```bash
git add scripts/discord_notify.py
git commit -m "feat: wire up main entry point for discord_notify"
```

---

## Task 7: Configure hooks and update .gitignore

**Files:**
- Create: `.claude/settings.local.json`
- Modify: `.gitignore`

- [ ] **Step 1: Add `.claude/settings.local.json` to `.gitignore`**

Open `.gitignore` and append:

```
.claude/settings.local.json
```

- [ ] **Step 2: Create `.claude/settings.local.json`**

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "mcp__linear__save_issue",
        "hooks": [
          {
            "type": "command",
            "command": "cd /Users/Privat/Projects/Raphael && .venv/bin/python scripts/discord_notify.py --mode done"
          }
        ]
      },
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "cd /Users/Privat/Projects/Raphael && .venv/bin/python scripts/discord_notify.py --mode push"
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 3: Verify the hooks file is gitignored**

```bash
git status
```

Expected: `.claude/settings.local.json` does NOT appear in untracked files.

- [ ] **Step 4: Commit .gitignore change**

```bash
git add .gitignore
git commit -m "chore: ignore .claude/settings.local.json (machine-specific hooks)"
```

---

## Task 8: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add Pipeline section to CLAUDE.md**

Find the `## Task Tracking` section in `CLAUDE.md` and insert the following section directly above it:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document automated Discord changelog pipeline in CLAUDE.md"
```

---

## Task 9: End-to-end verification

- [ ] **Step 1: Run full test suite and confirm everything passes**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: all tests PASS, 0 failures.

- [ ] **Step 2: Live test — Done hook**

In this Claude Code session, update a test Linear issue to Done and verify a message appears in the Discord channel.

- [ ] **Step 3: Live test — Push hook**

Make a trivial commit and push, then verify the changelog appears in Discord.

```bash
git commit --allow-empty -m "chore: verify Discord push hook"
git push origin dev
```

- [ ] **Step 4: Push all implementation commits**

```bash
git push origin dev
```

- [ ] **Step 5: Mark TEN-41 Done in Linear**
