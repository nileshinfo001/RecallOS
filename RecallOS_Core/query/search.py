"""Natural-language query parser and search for RecallOS."""

from __future__ import annotations

import re

from memory.embed import embed_text
from memory.faiss_db import MemoryDB

# Common stop-words / filler to strip when extracting the object name
_STOP = {
    "where", "are", "is", "my", "the", "a", "an", "of", "to", "in",
    "at", "on", "for", "me", "i", "can", "you", "find", "see", "did",
    "have", "has", "was", "were", "do", "does", "put", "left", "last",
    "seen", "locate", "please", "whats", "what's",
}


def extract_object(query: str) -> str:
    """Pull the object noun out of a casual query like 'where are my keys'."""
    # strip punctuation, lowercase
    cleaned = re.sub(r"[^\w\s]", " ", query.lower())
    tokens = [t for t in cleaned.split() if t and t not in _STOP]
    if not tokens:
        return query.strip().lower()
    # Prefer the last content token ("where are my car keys" → "keys")
    # but join all remaining for multi-word objects ("remote control")
    return " ".join(tokens)


def search_memory(query_text: str, db: MemoryDB, top_k: int = 3) -> list[dict]:
    """Extract object from query, embed, search FAISS, print results."""
    obj = extract_object(query_text)
    print(f"[query] Parsed object: '{obj}'  (from: \"{query_text}\")")

    emb = embed_text(obj)
    hits = db.search(emb, top_k=top_k)

    if not hits:
        print("[query] No memories found.")
        return []

    for hit in hits:
        print(
            f"Last seen: {hit['location']} at {hit['timestamp']} "
            f"- {hit['thumbnail_path']}  (score={hit['score']:.3f})"
        )
    return hits
