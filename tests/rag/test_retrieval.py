"""Tests for rag/retriever.py — retrieval quality against real ChromaDB index.

These tests require a populated ChromaDB index at data/chroma/.
They skip automatically if the index doesn't exist.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

CHROMA_DIR = Path(__file__).parent.parent.parent / "data" / "chroma"
requires_index = pytest.mark.skipif(
    not CHROMA_DIR.exists(),
    reason="ChromaDB index not found at data/chroma/ — run build_index.py first",
)


@requires_index
class TestBareEntityRetrieval:
    """Bare entity queries (1-3 words, no question starter) should get exact page matches."""

    def test_known_skill_returns_exact_page(self):
        from rag.retriever import query
        results = query("Predator?")
        assert len(results) > 0
        assert all(r["page_title"] == "Predator" for r in results)
        assert all(r["score"] == 1.0 for r in results)

    def test_known_multi_word_skill(self):
        from rag.retriever import query
        results = query("Great Sage?")
        assert len(results) > 0
        assert all(r["page_title"] == "Great Sage" for r in results)

    def test_known_race(self):
        from rag.retriever import query
        results = query("HihiIrokane Ingot?")
        assert len(results) > 0
        assert results[0]["page_title"] == "HihiIrokane Ingot"

    def test_blocked_overview_page_falls_through(self):
        """'Races?' should NOT return the Races overview page (it's blocklisted)."""
        from rag.retriever import query
        results = query("Races?")
        # Should fall through to semantic search — may return specific race pages
        for r in results:
            assert r["page_title"] != "Races", "Blocked overview page should not appear"


@requires_index
class TestSemanticRetrieval:
    """Structured questions should retrieve relevant pages via semantic search."""

    def test_skill_question_returns_relevant_page(self):
        from rag.retriever import query
        results = query("What does Great Sage do?")
        page_titles = [r["page_title"] for r in results]
        assert any("Great Sage" in t for t in page_titles)

    def test_race_question_returns_relevant_page(self):
        from rag.retriever import query
        results = query("What are the stats for the Kijin race?")
        page_titles = [r["page_title"] for r in results]
        assert any("Kijin" in t for t in page_titles)

    def test_item_question_returns_relevant_page(self):
        from rag.retriever import query
        results = query("What does the Battlewill Manual do?")
        page_titles = [r["page_title"] for r in results]
        assert any("Battlewill" in t for t in page_titles)

    def test_mob_question_returns_relevant_page(self):
        from rag.retriever import query
        results = query("Where does the Direwolf spawn?")
        page_titles = [r["page_title"] for r in results]
        assert any("Direwolf" in t for t in page_titles)

    def test_garbage_query_returns_few_low_quality_results(self):
        from rag.retriever import query
        results = query("zxqwerty nonsense query 999?")
        # Embedding models find *something* for any input, but results should
        # be sparse and low-scoring compared to real queries
        assert len(results) <= 4, f"Garbage query returned {len(results)} results — expected few"


@requires_index
class TestBlocklistFiltering:
    """Generic overview pages should never appear in results."""

    def test_wiki_welcome_excluded(self):
        from rag.retriever import query
        results = query("How do I play the Tensura mod?")
        for r in results:
            assert not r["page_title"].startswith("Tensura: Reincarnated Wiki"), \
                f"Wiki meta page {r['page_title']} should be blocked"

    def test_generic_mobs_page_excluded(self):
        from rag.retriever import query
        results = query("What mobs are in the mod?")
        for r in results:
            assert r["page_title"] != "Mobs", "Generic Mobs overview should be blocked"

    def test_specific_mob_subpage_allowed(self):
        from rag.retriever import query
        results = query("Direwolf?")
        page_titles = [r["page_title"] for r in results]
        # Mobs/Direwolf should be allowed (subpages of exact-blocked pages are fine)
        assert any("Direwolf" in t for t in page_titles)


@requires_index
class TestPageTitleBoost:
    """TEN-113: Entity names in structured questions should boost matching pages."""

    def test_predator_mechanics_includes_predator_page(self):
        """Page title boost should inject 'Predator' as a variant for structured queries."""
        from rag.retriever import query
        results = query("Can predator copy unique skills?")
        page_titles = [r["page_title"] for r in results]
        assert any("Predator" in t for t in page_titles), \
            f"Predator page should appear — got: {page_titles}"

    def test_haki_question_includes_haki_page(self):
        from rag.retriever import query
        results = query("How do you get haki?")
        page_titles = [r["page_title"] for r in results]
        assert any("Haki" in t for t in page_titles)

    def test_predator_mechanics_includes_predator_page(self):
        from rag.retriever import query
        results = query("Can predator copy unique skills?")
        page_titles = [r["page_title"] for r in results]
        assert any("Predator" in t for t in page_titles)


@requires_index
class TestEntityFallback:
    """TEN-111: When bare entity lookup fails, entity name should be in semantic variants."""

    def test_entity_present_in_results_even_without_exact_match(self):
        """Shadow Striker has a page — even if bare lookup glitches, semantic should find it."""
        from rag.retriever import query
        results = query("Shadow Striker?")
        page_titles = [r["page_title"] for r in results]
        assert any("Shadow Striker" in t for t in page_titles)


# ---------------------------------------------------------------------------
# Phase 3 — Enumeration detection (unit tests, no ChromaDB needed)
# ---------------------------------------------------------------------------

class TestEnumerationDetection:
    """TEN-64: Detect enumeration intent and map to category strategies."""

    def test_list_all_races(self):
        from rag.retriever import _detect_enumeration
        assert _detect_enumeration("list all races?") == "prefix:Races"

    def test_list_all_unique_skills(self):
        from rag.retriever import _detect_enumeration
        assert _detect_enumeration("list all unique skills?") == "content:Unique Skill"

    def test_what_extra_skills_exist(self):
        from rag.retriever import _detect_enumeration
        assert _detect_enumeration("what extra skills are there?") == "content:Extra Skill"

    def test_show_every_common_skill(self):
        from rag.retriever import _detect_enumeration
        assert _detect_enumeration("show every common skill?") == "content:Common Skill"

    def test_list_all_magics(self):
        from rag.retriever import _detect_enumeration
        assert _detect_enumeration("list all magics?") == "prefix:Abilities/Magics"

    def test_how_many_mobs(self):
        from rag.retriever import _detect_enumeration
        assert _detect_enumeration("how many mobs are there?") == "prefix:Mobs"

    def test_list_all_battlewills(self):
        from rag.retriever import _detect_enumeration
        assert _detect_enumeration("list all battlewill skills?") == "content:Battlewill"

    def test_normal_query_not_detected(self):
        from rag.retriever import _detect_enumeration
        assert _detect_enumeration("What does Predator do?") is None

    def test_bare_entity_not_detected(self):
        from rag.retriever import _detect_enumeration
        assert _detect_enumeration("Predator?") is None

    def test_comparative_not_detected_as_enumeration(self):
        from rag.retriever import _detect_enumeration
        assert _detect_enumeration("What is the best race?") is None


class TestCategoryDetection:
    """_detect_category maps category terms to retrieval strategies."""

    def test_races(self):
        from rag.retriever import _detect_category
        assert _detect_category("strongest race?") == "prefix:Races"

    def test_unique_skills(self):
        from rag.retriever import _detect_category
        assert _detect_category("best unique skill?") == "content:Unique Skill"

    def test_broad_excluded_in_comparative(self):
        from rag.retriever import _detect_category
        assert _detect_category("best item?", exclude_broad=True) is None
        assert _detect_category("strongest mob?", exclude_broad=True) is None
        assert _detect_category("best block?", exclude_broad=True) is None

    def test_broad_allowed_without_exclusion(self):
        from rag.retriever import _detect_category
        assert _detect_category("best item?") == "prefix:Items"
        assert _detect_category("strongest mob?") == "prefix:Mobs"

    def test_no_category(self):
        from rag.retriever import _detect_category
        assert _detect_category("how do I play?") is None


class TestComparativeClassifier:
    """TEN-25: is_comparative detects comparison/recommendation intent."""

    def test_best_triggers(self):
        from rag.retriever import is_comparative
        assert is_comparative("What is the best race?") is True

    def test_strongest_triggers(self):
        from rag.retriever import is_comparative
        assert is_comparative("What is the strongest skill?") is True

    def test_compare_triggers(self):
        from rag.retriever import is_comparative
        assert is_comparative("Compare slime vs goblin?") is True

    def test_which_triggers(self):
        from rag.retriever import is_comparative
        assert is_comparative("Which race is better?") is True

    def test_worst_triggers(self):
        from rag.retriever import is_comparative
        assert is_comparative("What is the worst skill?") is True

    def test_normal_question_not_comparative(self):
        from rag.retriever import is_comparative
        assert is_comparative("What does Predator do?") is False

    def test_bare_entity_not_comparative(self):
        from rag.retriever import is_comparative
        assert is_comparative("Predator?") is False


class TestSynonymNormalization:
    """TEN-166: _normalize_query maps race-context phrases to wiki vocabulary."""

    def test_lesser_demon_to_daemon(self):
        from rag.retriever import _normalize_query
        assert _normalize_query("lesser demon?") == "lesser daemon?"

    def test_preserves_capitalization(self):
        from rag.retriever import _normalize_query
        assert _normalize_query("Lesser Demon?") == "Lesser Daemon?"

    def test_arch_demon(self):
        from rag.retriever import _normalize_query
        assert _normalize_query("arch demon?") == "arch daemon?"

    def test_no_change_for_unmapped(self):
        from rag.retriever import _normalize_query
        assert _normalize_query("Predator?") == "Predator?"

    def test_single_demon_unchanged(self):
        """Single 'demon' is NOT mapped — preserves Demon Essence, Demon Lord Haki etc."""
        from rag.retriever import _normalize_query
        assert _normalize_query("Demon Essence?") == "Demon Essence?"

    def test_demon_dominate_unchanged(self):
        from rag.retriever import _normalize_query
        assert _normalize_query("demon dominate?") == "demon dominate?"


class TestVersionBlocklist:
    """TEN-165: Version/changelog pages are blocked."""

    def test_version_page_blocked(self):
        from rag.retriever import _is_blocked
        assert _is_blocked("1.19.2 1.0.0.0") is True
        assert _is_blocked("1.19.2 1.0.0.1") is True

    def test_normal_page_not_blocked(self):
        from rag.retriever import _is_blocked
        assert _is_blocked("Predator") is False
        assert _is_blocked("Races/Human") is False


# ---------------------------------------------------------------------------
# Phase 3 — Enumeration retrieval (requires ChromaDB)
# ---------------------------------------------------------------------------

@requires_index
class TestEnumerationRetrieval:
    """TEN-64/TEN-50: Category retrieval returns complete lists."""

    def test_list_all_races_returns_many(self):
        from rag.retriever import query
        results = query("list all races?")
        assert len(results) >= 50, f"Expected 50+ races, got {len(results)}"
        assert all(r["page_title"].startswith("Races/") for r in results)

    def test_list_all_unique_skills_returns_many(self):
        from rag.retriever import query
        results = query("list all unique skills?")
        assert len(results) >= 40, f"Expected 40+ unique skills, got {len(results)}"

    def test_list_all_magics_returns_many(self):
        from rag.retriever import query
        results = query("list all magics?")
        assert len(results) >= 20, f"Expected 20+ magics, got {len(results)}"

    def test_enumeration_fits_token_budget(self):
        """All enumeration results should fit within MAX_ENUM_CHARS."""
        from rag.retriever import query, MAX_ENUM_CHARS
        results = query("list all races?")
        total = sum(len(r["text"]) for r in results)
        assert total <= MAX_ENUM_CHARS, f"Enumeration {total} chars exceeds {MAX_ENUM_CHARS}"

    def test_normal_query_not_affected_by_enumeration(self):
        from rag.retriever import query
        results = query("Predator?")
        assert all(r["page_title"] == "Predator" for r in results)


@requires_index
class TestComparativeRetrieval:
    """TEN-25: Comparative queries about categories get full data with verbose stats."""

    def test_strongest_race_gets_all_races(self):
        from rag.retriever import query
        results = query("what is the strongest race?")
        assert len(results) >= 40, f"Expected 40+ races for comparison, got {len(results)}"

    def test_verbose_includes_attack_stats(self):
        from rag.retriever import query
        results = query("what is the strongest race?")
        all_text = " ".join(r["text"] for r in results)
        assert "Attack DMG:" in all_text, "Verbose mode should include Attack DMG"
        assert "Intrinsics:" in all_text or "Previous:" in all_text, \
            "Verbose mode should include Intrinsics or evolution chain"

    def test_best_unique_skill_gets_all_skills(self):
        from rag.retriever import query
        results = query("what is the best unique skill?")
        assert len(results) >= 40, f"Expected 40+ skills for comparison, got {len(results)}"

    def test_broad_category_comparative_uses_semantic(self):
        """'best item?' should NOT dump all items — falls through to semantic search."""
        from rag.retriever import query
        results = query("what is the best item?")
        assert len(results) <= 10, f"Broad category should use semantic K=10, got {len(results)}"
