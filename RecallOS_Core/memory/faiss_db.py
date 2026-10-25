"""FAISS-backed persistent memory store for RecallOS."""

from __future__ import annotations

import json
import os
from typing import Any

import faiss
import numpy as np

DIM = 384
DEFAULT_DIR = os.path.join(".", "data")


class MemoryDB:
    """Stores object sightings with FAISS cosine (inner-product on L2-normed vectors)."""

    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or DEFAULT_DIR
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index_path = os.path.join(self.persist_dir, "index.faiss")
        self.meta_path = os.path.join(self.persist_dir, "metadata.json")

        self.metadata: list[dict[str, Any]] = []
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(DIM)

        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r") as f:
                self.metadata = json.load(f)
            print(f"[db] Loaded {len(self.metadata)} memories from disk")
        else:
            print("[db] Fresh memory store")

    def save(self) -> None:
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"[db] Persisted {len(self.metadata)} memories")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, item: dict[str, Any]) -> None:
        """Add a memory item.

        Expected keys: object_name, timestamp, location, embedding, thumbnail_path
        """
        emb = np.asarray(item["embedding"], dtype=np.float32).reshape(1, -1)
        if emb.shape[1] != DIM:
            raise ValueError(f"Expected {DIM}-dim embedding, got {emb.shape[1]}")

        # Ensure L2-normalized for cosine via inner product
        faiss.normalize_L2(emb)
        self.index.add(emb)

        self.metadata.append(
            {
                "object_name": item["object_name"],
                "timestamp": item["timestamp"],
                "location": item.get("location", "bedroom"),
                "thumbnail_path": item.get("thumbnail_path", ""),
            }
        )
        print(f"[db] Added: {item['object_name']} @ {item.get('location', 'bedroom')}")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[dict[str, Any]]:
        """Cosine-similarity search. Returns top_k metadata dicts with 'score'."""
        if self.index.ntotal == 0:
            return []

        q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = dict(self.metadata[idx])
            entry["score"] = float(score)
            results.append(entry)
        return results

    def __len__(self) -> int:
        return self.index.ntotal
