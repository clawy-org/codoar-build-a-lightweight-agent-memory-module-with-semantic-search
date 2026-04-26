#!/usr/bin/env python3
"""
test_agent_memory.py — Comprehensive tests for AgentMemory.

Covers: add, search, dedup, forget, list, stats, persistence.
Run with: python -m pytest test_agent_memory.py -v
"""

import json
import os
import tempfile

import pytest

from agent_memory import AgentMemory, _cosine_similarity


@pytest.fixture
def tmp_path_file():
    """Provide a temporary JSON file path, cleaned up after test."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    # Remove the file so AgentMemory starts fresh
    os.remove(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def memory(tmp_path_file):
    """Provide a fresh AgentMemory instance."""
    return AgentMemory(path=tmp_path_file)


# ── Add ──────────────────────────────────────────────────────────────

class TestAdd:
    def test_add_returns_id(self, memory):
        mid = memory.add("Hello world")
        assert mid is not None
        assert isinstance(mid, str)
        assert len(mid) > 0

    def test_add_with_tags(self, memory):
        mid = memory.add("User likes tea", tags=["preferences", "drinks"])
        assert mid is not None
        mems = memory.list(tag="preferences")
        assert len(mems) == 1
        assert mems[0]["text"] == "User likes tea"
        assert "drinks" in mems[0]["tags"]

    def test_add_without_tags(self, memory):
        mid = memory.add("A fact without tags")
        assert mid is not None
        mems = memory.list()
        assert len(mems) == 1
        assert mems[0]["tags"] == []

    def test_add_persists_to_disk(self, memory, tmp_path_file):
        memory.add("Persistent memory")
        assert os.path.exists(tmp_path_file)
        with open(tmp_path_file, "r") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["text"] == "Persistent memory"

    def test_add_multiple(self, memory):
        memory.add("First")
        memory.add("Second")
        memory.add("Third")
        assert memory.stats()["count"] == 3


# ── Deduplication ────────────────────────────────────────────────────

class TestDedup:
    def test_exact_duplicate_rejected(self, memory):
        mid1 = memory.add("User prefers dark mode")
        mid2 = memory.add("User prefers dark mode")
        assert mid1 is not None
        assert mid2 is None
        assert memory.stats()["count"] == 1

    def test_near_duplicate_rejected(self, memory):
        mid1 = memory.add("User prefers dark mode in all apps")
        mid2 = memory.add("User prefers dark mode for all applications")
        assert mid1 is not None
        assert mid2 is None
        assert memory.stats()["count"] == 1

    def test_different_content_accepted(self, memory):
        mid1 = memory.add("User prefers dark mode")
        mid2 = memory.add("Meeting scheduled for Friday at 3pm")
        assert mid1 is not None
        assert mid2 is not None
        assert memory.stats()["count"] == 2


# ── Search ───────────────────────────────────────────────────────────

class TestSearch:
    def test_search_empty_memory(self, memory):
        results = memory.search("anything")
        assert results == []

    def test_search_returns_relevant(self, memory):
        memory.add("User prefers dark mode", tags=["preferences"])
        memory.add("Meeting on Friday at 3pm", tags=["calendar"])
        memory.add("User works in AI research", tags=["work"])

        results = memory.search("what theme does the user want?", top_k=1)
        assert len(results) == 1
        assert "dark mode" in results[0]["text"]

    def test_search_respects_top_k(self, memory):
        for i in range(10):
            memory.add(f"Memory number {i} about topic {i * 7}")
        results = memory.search("topic", top_k=3)
        assert len(results) == 3

    def test_search_results_have_similarity(self, memory):
        memory.add("Python is great")
        results = memory.search("programming language", top_k=1)
        assert len(results) == 1
        assert "similarity" in results[0]
        assert 0 <= results[0]["similarity"] <= 1

    def test_search_results_sorted_by_similarity(self, memory):
        memory.add("The weather is sunny today")
        memory.add("Python programming language")
        memory.add("Machine learning algorithms")

        results = memory.search("coding in Python", top_k=3)
        sims = [r["similarity"] for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_search_results_exclude_embedding(self, memory):
        memory.add("Test content")
        results = memory.search("test", top_k=1)
        assert "embedding" not in results[0]


# ── Forget ───────────────────────────────────────────────────────────

class TestForget:
    def test_forget_existing(self, memory):
        mid = memory.add("To be forgotten")
        assert memory.stats()["count"] == 1
        result = memory.forget(mid)
        assert result is True
        assert memory.stats()["count"] == 0

    def test_forget_nonexistent(self, memory):
        result = memory.forget("nonexistent-id")
        assert result is False

    def test_forget_persists(self, memory, tmp_path_file):
        mid = memory.add("Will be removed")
        memory.forget(mid)
        # Reload from disk
        mem2 = AgentMemory(path=tmp_path_file)
        assert mem2.stats()["count"] == 0

    def test_forget_only_target(self, memory):
        mid1 = memory.add("Keep this")
        mid2 = memory.add("Remove this one about a totally different subject")
        memory.forget(mid2)
        assert memory.stats()["count"] == 1
        remaining = memory.list()
        assert remaining[0]["id"] == mid1


# ── List ─────────────────────────────────────────────────────────────

class TestList:
    def test_list_empty(self, memory):
        assert memory.list() == []

    def test_list_all(self, memory):
        memory.add("A")
        memory.add("B is about something completely different from A")
        mems = memory.list()
        assert len(mems) == 2

    def test_list_filter_by_tag(self, memory):
        memory.add("Python stuff", tags=["coding"])
        memory.add("Weather update for tomorrow morning", tags=["weather"])
        memory.add("More coding with JavaScript frameworks", tags=["coding"])

        coding = memory.list(tag="coding")
        assert len(coding) == 2
        for m in coding:
            assert "coding" in m["tags"]

    def test_list_no_match(self, memory):
        memory.add("Hello", tags=["greeting"])
        result = memory.list(tag="nonexistent")
        assert result == []

    def test_list_excludes_embedding(self, memory):
        memory.add("Test")
        mems = memory.list()
        assert "embedding" not in mems[0]


# ── Stats ────────────────────────────────────────────────────────────

class TestStats:
    def test_stats_empty(self, memory):
        s = memory.stats()
        assert s["count"] == 0
        assert s["total_size_bytes"] == 0
        assert s["oldest"] is None
        assert s["newest"] is None

    def test_stats_with_data(self, memory):
        memory.add("First memory")
        memory.add("Second memory about a very different topic entirely")
        s = memory.stats()
        assert s["count"] == 2
        assert s["total_size_bytes"] > 0
        assert s["oldest"] is not None
        assert s["newest"] is not None
        assert s["oldest"] <= s["newest"]


# ── Persistence ──────────────────────────────────────────────────────

class TestPersistence:
    def test_roundtrip(self, tmp_path_file):
        mem1 = AgentMemory(path=tmp_path_file)
        mid = mem1.add("Persistent fact", tags=["test"])

        # Create new instance from same file
        mem2 = AgentMemory(path=tmp_path_file)
        assert mem2.stats()["count"] == 1
        mems = mem2.list()
        assert mems[0]["text"] == "Persistent fact"
        assert mems[0]["id"] == mid

    def test_search_after_reload(self, tmp_path_file):
        mem1 = AgentMemory(path=tmp_path_file)
        mem1.add("User prefers vim over emacs", tags=["preferences"])

        mem2 = AgentMemory(path=tmp_path_file)
        results = mem2.search("text editor preference", top_k=1)
        assert len(results) == 1
        assert "vim" in results[0]["text"]

    def test_corrupted_file(self, tmp_path_file):
        # Write invalid JSON
        with open(tmp_path_file, "w") as f:
            f.write("{{not valid json")

        mem = AgentMemory(path=tmp_path_file)
        assert mem.stats()["count"] == 0

    def test_nonexistent_file(self):
        mem = AgentMemory(path="/tmp/does_not_exist_test_12345.json")
        assert mem.stats()["count"] == 0
        if os.path.exists("/tmp/does_not_exist_test_12345.json"):
            os.remove("/tmp/does_not_exist_test_12345.json")


# ── Cosine similarity helper ────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert _cosine_similarity(a, b) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
