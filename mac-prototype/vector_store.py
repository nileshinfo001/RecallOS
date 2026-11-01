"""FAISS IndexFlatIP store for L2-normalized 384-d visual embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

DIM = 384
ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "faiss.index"
META_PATH = ROOT / "metadata.json"


class VectorStore:
    def __init__(
        self,
        index_path: str | Path | None = None,
        meta_path: str | Path | None = None,
    ):
        self.index_path = Path(index_path) if index_path else INDEX_PATH
        self.meta_path = Path(meta_path) if meta_path else META_PATH
        self.records: list[dict[str, Any]] = []
        self.index = faiss.IndexFlatIP(DIM)
        self.load()

    def add(self, id: str, embedding: np.ndarray, metadata: dict) -> None:
        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if emb.shape[1] != DIM:
            raise ValueError(f"Expected {DIM}-dim embedding, got {emb.shape[1]}")
        faiss.normalize_L2(emb)
        self.index.add(emb)
        record = {"id": id}
        record.update(metadata)
        self.records.append(record)
        self.save()

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
        threshold: float = 0.65,
    ) -> list[dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(q, k)

        hits: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            sim = float(score)
            if sim < threshold:
                continue
            entry = dict(self.records[int(idx)])
            entry["score"] = sim
            hits.append(entry)
        return hits

    def save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.records, indent=2))

    def load(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.records = json.loads(self.meta_path.read_text())
            print(f"[store] Loaded {len(self.records)} embeddings")
        else:
            self.index = faiss.IndexFlatIP(DIM)
            self.records = []
            print("[store] Fresh index")


def get_store() -> VectorStore:
    return VectorStore()
