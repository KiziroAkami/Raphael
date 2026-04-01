import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# Make sure the project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.discord_notify import categorise_commit, first_sentence, _parse_response


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
