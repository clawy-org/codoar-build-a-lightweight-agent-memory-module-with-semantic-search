# Agent Memory — Lightweight Semantic Search

A self-contained Python module for AI agent long-term memory. Store, search, and retrieve facts using local semantic embeddings — no LLM API calls required.

## Features

- **Semantic search** via `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Automatic deduplication** — rejects memories with >0.95 cosine similarity
- **JSON persistence** — human-readable, file-based storage
- **Tag filtering** — organize memories with arbitrary tags
- **Zero API calls** — works fully offline after initial model download

## Installation

```bash
pip install sentence-transformers
```

The embedding model (`all-MiniLM-L6-v2`, ~80MB) downloads automatically on first use.

## Quick Start

```python
from agent_memory import AgentMemory

memory = AgentMemory(path="memory.json")

# Add memories
memory.add("User prefers dark mode", tags=["preferences", "ui"])
memory.add("User works in AI research", tags=["identity", "work"])
memory.add("Meeting with team on Friday at 3pm", tags=["calendar"])

# Semantic search
results = memory.search("what does the user like about interfaces?", top_k=5)
for r in results:
    print(f"  [{r['similarity']:.4f}] {r['text']}")
# → [0.5823] User prefers dark mode

# Filter by tag
prefs = memory.list(tag="preferences")

# Delete a memory
memory.forget(results[0]["id"])

# Stats
print(memory.stats())
# → {'count': 2, 'total_size_bytes': 4521, 'oldest': 1712345678.9, 'newest': 1712345680.1}
```

## API Reference

### `AgentMemory(path="memory.json")`

Create or load a memory store. If the file exists, memories are loaded automatically.

### `.add(text, tags=None) → str | None`

Add a memory. Returns the memory ID, or `None` if rejected as a near-duplicate (>0.95 cosine similarity to an existing memory). Auto-saves after each write.

### `.search(query, top_k=5) → list[dict]`

Semantic search. Returns up to `top_k` results sorted by cosine similarity. Each result contains:
- `id`, `text`, `tags`, `created_at`, `similarity`

### `.forget(memory_id) → bool`

Delete a memory by ID. Returns `True` if found and removed.

### `.list(tag=None) → list[dict]`

List all memories, optionally filtered by a tag. Results contain `id`, `text`, `tags`, `created_at`.

### `.stats() → dict`

Returns `count`, `total_size_bytes`, `oldest` timestamp, `newest` timestamp.

## Storage Format

Memories are stored as a JSON array:

```json
[
  {
    "id": "uuid-string",
    "text": "User prefers dark mode",
    "tags": ["preferences", "ui"],
    "created_at": 1712345678.9,
    "embedding": [0.012, -0.034, ...]
  }
]
```

## Running Tests

```bash
pip install pytest
python -m pytest test_agent_memory.py -v
```

## Demo

```bash
python agent_memory.py
```

Sample output:

```
============================================================
  AgentMemory — Demo
============================================================

📝 Adding memories...
  ✅ id=a1b2c3d4...  User prefers dark mode in all applications
  ✅ id=e5f6a7b8...  User's name is Misha
  ✅ id=c9d0e1f2...  User works in AI/ML research
  ...

🔍 Testing deduplication...
  Near-duplicate result: rejected ✅

🔎 Searching: 'what does the user like?'
  [0.5432] User prefers dark mode in all applications
  [0.4567] User likes Python and TypeScript
  [0.3210] User works in AI/ML research

📊 Stats:
  Memories: 6
  Size: 12,345 bytes
```

## Requirements

- Python 3.9+
- `sentence-transformers` (installs `torch`, `transformers`, etc.)
- Works offline after initial model download
