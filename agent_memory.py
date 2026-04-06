#!/usr/bin/env python3
"""
agent_memory.py — Lightweight agent memory module with semantic search.

A self-contained module for agent long-term memory: store, search, and
retrieve facts using local semantic embeddings (sentence-transformers
with all-MiniLM-L6-v2). No LLM API calls required.

Usage as library:
    from agent_memory import AgentMemory

    memory = AgentMemory(path="memory.json")
    memory.add("User prefers dark mode", tags=["preferences", "ui"])
    results = memory.search("what does the user like about interfaces?", top_k=5)
    memory.forget(results[0]["id"])
    memory.list(tag="preferences")
    memory.stats()

Usage as demo:
    python agent_memory.py
"""

import json
import math
import os
import time
import uuid
from typing import Any, Dict, List, Optional

# Lazy-loaded to keep import fast when not searching
_model = None


def _get_model():
    """Lazy-load the sentence-transformers model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class AgentMemory:
    """
    A persistent, semantic-search-enabled memory store for AI agents.

    Memories are stored as JSON on disk. Each memory has an id, text,
    optional tags, a creation timestamp, and an embedding vector for
    semantic search.

    Deduplication: new memories with >0.95 cosine similarity to an
    existing memory are rejected.
    """

    def __init__(self, path: str = "memory.json"):
        """
        Initialize memory, loading from disk if the file exists.

        Args:
            path: Path to the JSON file for persistence.
        """
        self.path = path
        self._memories: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load memories from disk if the file exists."""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._memories = data
                else:
                    self._memories = []
            except (json.JSONDecodeError, IOError):
                self._memories = []

    def _save(self) -> None:
        """Persist memories to disk as JSON."""
        # Ensure parent directory exists
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._memories, f, indent=2, ensure_ascii=False)

    def _embed(self, text: str) -> List[float]:
        """Generate an embedding for the given text."""
        model = _get_model()
        embedding = model.encode([text])[0]
        return embedding.tolist()

    def _is_duplicate(self, embedding: List[float], threshold: float = 0.95) -> bool:
        """Check if an embedding is too similar to any existing memory."""
        for mem in self._memories:
            if "embedding" in mem:
                sim = _cosine_similarity(embedding, mem["embedding"])
                if sim > threshold:
                    return True
        return False

    def add(self, text: str, tags: Optional[List[str]] = None) -> Optional[str]:
        """
        Add a memory. Returns the memory ID, or None if rejected as duplicate.

        Args:
            text: The memory content.
            tags: Optional list of tags for categorization.

        Returns:
            The memory ID string, or None if deduplicated.
        """
        embedding = self._embed(text)

        if self._is_duplicate(embedding):
            return None

        memory_id = str(uuid.uuid4())
        memory = {
            "id": memory_id,
            "text": text,
            "tags": tags or [],
            "created_at": time.time(),
            "embedding": embedding,
        }
        self._memories.append(memory)
        self._save()
        return memory_id

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search over memories.

        Args:
            query: Natural language query.
            top_k: Maximum number of results to return.

        Returns:
            List of memory dicts (without embeddings), sorted by relevance.
        """
        if not self._memories:
            return []

        query_embedding = self._embed(query)

        scored = []
        for mem in self._memories:
            if "embedding" not in mem:
                continue
            sim = _cosine_similarity(query_embedding, mem["embedding"])
            scored.append((sim, mem))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sim, mem in scored[:top_k]:
            result = {
                "id": mem["id"],
                "text": mem["text"],
                "tags": mem["tags"],
                "created_at": mem["created_at"],
                "similarity": round(sim, 4),
            }
            results.append(result)
        return results

    def forget(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.

        Args:
            memory_id: The ID of the memory to remove.

        Returns:
            True if the memory was found and removed, False otherwise.
        """
        before = len(self._memories)
        self._memories = [m for m in self._memories if m["id"] != memory_id]
        removed = len(self._memories) < before
        if removed:
            self._save()
        return removed

    def list(self, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List memories, optionally filtered by tag.

        Args:
            tag: If provided, only return memories with this tag.

        Returns:
            List of memory dicts (without embeddings).
        """
        results = []
        for mem in self._memories:
            if tag and tag not in mem.get("tags", []):
                continue
            results.append({
                "id": mem["id"],
                "text": mem["text"],
                "tags": mem["tags"],
                "created_at": mem["created_at"],
            })
        return results

    def stats(self) -> Dict[str, Any]:
        """
        Return memory statistics.

        Returns:
            Dict with count, total_size_bytes, oldest and newest timestamps.
        """
        if not self._memories:
            return {
                "count": 0,
                "total_size_bytes": 0,
                "oldest": None,
                "newest": None,
            }

        timestamps = [m["created_at"] for m in self._memories]
        # Approximate size from JSON serialization
        size = len(json.dumps(self._memories, ensure_ascii=False).encode("utf-8"))

        return {
            "count": len(self._memories),
            "total_size_bytes": size,
            "oldest": min(timestamps),
            "newest": max(timestamps),
        }


# ── Demo ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("  AgentMemory — Demo")
    print("=" * 60)

    # Use a temp file so we don't clutter the workspace
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        demo_path = tmp.name

    try:
        mem = AgentMemory(path=demo_path)

        # Add some memories
        print("\n📝 Adding memories...")
        memories_to_add = [
            ("User prefers dark mode in all applications", ["preferences", "ui"]),
            ("User's name is Misha", ["identity"]),
            ("User works in AI/ML research", ["identity", "work"]),
            ("User timezone is America/Los_Angeles", ["preferences", "location"]),
            ("User likes Python and TypeScript", ["preferences", "programming"]),
            ("Meeting with team scheduled for Friday at 3pm", ["calendar"]),
        ]

        for text, tags in memories_to_add:
            mid = mem.add(text, tags=tags)
            status = f"✅ id={mid[:8]}..." if mid else "⚠️  duplicate, skipped"
            print(f"  {status}  {text}")

        # Try adding a near-duplicate
        print("\n🔍 Testing deduplication...")
        dup_id = mem.add("User prefers dark mode for all apps", tags=["preferences"])
        print(f"  Near-duplicate result: {'rejected ✅' if dup_id is None else 'added (unexpected)'}")

        # Semantic search
        print("\n🔎 Searching: 'what does the user like?'")
        results = mem.search("what does the user like?", top_k=3)
        for r in results:
            print(f"  [{r['similarity']:.4f}] {r['text']}")

        print("\n🔎 Searching: 'when is the meeting?'")
        results = mem.search("when is the meeting?", top_k=2)
        for r in results:
            print(f"  [{r['similarity']:.4f}] {r['text']}")

        # Filter by tag
        print("\n🏷️  Listing memories with tag='preferences':")
        prefs = mem.list(tag="preferences")
        for p in prefs:
            print(f"  • {p['text']}  (tags: {p['tags']})")

        # Stats
        print("\n📊 Stats:")
        stats = mem.stats()
        print(f"  Memories: {stats['count']}")
        print(f"  Size: {stats['total_size_bytes']:,} bytes")

        # Forget
        print("\n🗑️  Forgetting first memory...")
        all_mems = mem.list()
        if all_mems:
            forgotten = mem.forget(all_mems[0]["id"])
            print(f"  Removed: {forgotten}")
            print(f"  Memories remaining: {mem.stats()['count']}")

        print("\n" + "=" * 60)
        print("  Demo complete!")
        print("=" * 60)

    finally:
        # Clean up temp file
        if os.path.exists(demo_path):
            os.remove(demo_path)
