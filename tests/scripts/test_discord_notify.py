import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# Make sure the project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.discord_notify import categorise_commit, first_sentence, _parse_response
from scripts.discord_notify import handle_done


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


class TestHandleDone:
    def _payload(self, status: str = "Done", issue_id: str = "TEN-41",
                 title: str = "Add Discord hooks", description: str = "Some description.",
                 milestone: str | None = None) -> dict:
        tool_response: dict = {
            "id": issue_id,
            "title": title,
            "status": status,
            "description": description,
        }
        if milestone is not None:
            tool_response["projectMilestone"] = {"name": milestone}
        return {
            "tool_name": "mcp__linear__save_issue",
            "tool_input": {"id": issue_id, "state": status},
            "tool_response": tool_response,
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
    @patch("scripts.discord_notify.groq_summarise", return_value="Added PostToolUse hooks for Discord.")
    def test_includes_milestone_when_present(self, mock_groq, mock_post):
        handle_done(self._payload(milestone="Phase 2 — Hardening"))
        msg = mock_post.call_args[0][0]
        assert "Phase 2 — Hardening" in msg

    @patch("scripts.discord_notify.post_discord")
    @patch("scripts.discord_notify.groq_summarise", return_value="Added PostToolUse hooks for Discord.")
    def test_omits_milestone_when_absent(self, mock_groq, mock_post):
        handle_done(self._payload())
        msg = mock_post.call_args[0][0]
        assert "`" not in msg.split("\n")[0]  # no backtick label on the header line

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
