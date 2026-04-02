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
