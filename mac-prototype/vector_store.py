"""Two FAISS indexes: DINOv2-small (384) and DINOv2-base (768)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

SMALL_DIM = 384
BASE_DIM = 768
ROOT = Path(__file__).resolve().parent
SMALL_INDEX_PATH = ROOT / "faiss_small.index"
BASE_INDEX_PATH = ROOT / "faiss_base.index"
META_PATH = ROOT / "metadata.json"


def _faiss():
    """Import FAISS after YOLO. Loading both earlier segfaults on this Mac."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import faiss
    return faiss


class VectorStore:
    def __init__(
        self,
        small_index_path: str | Path | None = None,
        base_index_path: str | Path | None = None,
        meta_path: str | Path | None = None,
    ):
        self.small_index_path = Path(small_index_path) if small_index_path else SMALL_INDEX_PATH
        self.base_index_path = Path(base_index_path) if base_index_path else BASE_INDEX_PATH
        self.meta_path = Path(meta_path) if meta_path else META_PATH
        faiss = _faiss()
        self.small_index = faiss.IndexFlatIP(SMALL_DIM)
        self.base_index = faiss.IndexFlatIP(BASE_DIM)
        self.small_records: list[dict[str, Any]] = []
        self.base_records: list[dict[str, Any]] = []
        self.load()

    def _bucket(self, dim: int):
        if dim == SMALL_DIM:
            return self.small_index, self.small_records
        if dim == BASE_DIM:
            return self.base_index, self.base_records
        raise ValueError(f"Unsupported embedding dim {dim}")

    def add(self, id: str, embedding: np.ndarray, metadata: dict) -> None:
        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        index, records = self._bucket(emb.shape[1])
        _faiss().normalize_L2(emb)
        index.add(emb)
        record = {"id": id, "dim": int(emb.shape[1])}
        record.update(metadata)
        record["dim"] = int(emb.shape[1])
        image_path = record.get("image_path")
        if image_path:
            record["id"] = Path(str(image_path)).stem.lower()
        records.append(record)
        self.save()

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
        threshold: float = 0.65,
        apply_threshold: bool = True,
    ) -> list[dict[str, Any]]:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        index, records = self._bucket(query.shape[1])
        if index.ntotal == 0:
            return []
        _faiss().normalize_L2(query)
        count = min(k, index.ntotal)
        scores, indices = index.search(query, count)
        hits: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            sim = float(score)
            if apply_threshold and sim < threshold:
                continue
            entry = dict(records[int(idx)])
            entry["score"] = sim
            hits.append(entry)
        return hits

    def _without_label(self, records: list[dict[str, Any]], index, dim: int, label: str):
        """Drop every vector whose label matches, and rebuild that FAISS index."""
        faiss = _faiss()
        kept_records: list[dict[str, Any]] = []
        vectors: list[np.ndarray] = []
        removed = 0
        for i, record in enumerate(records):
            if str(record.get("label", "")).strip().lower() == label:
                removed += 1
                continue
            kept_records.append(record)
            vectors.append(np.asarray(index.reconstruct(int(i)), dtype=np.float32))
        rebuilt = faiss.IndexFlatIP(dim)
        if vectors:
            matrix = np.stack(vectors, axis=0).astype(np.float32)
            faiss.normalize_L2(matrix)
            rebuilt.add(matrix)
        return kept_records, rebuilt, removed

    def drop_label(self, label: str) -> int:
        """Remove a label from both indexes. One centroid per remaining label."""
        target = label.strip().lower()
        self.small_records, self.small_index, small_removed = self._without_label(
            self.small_records, self.small_index, SMALL_DIM, target
        )
        self.base_records, self.base_index, base_removed = self._without_label(
            self.base_records, self.base_index, BASE_DIM, target
        )
        self.save()
        return small_removed + base_removed

    def save(self) -> None:
        faiss = _faiss()
        faiss.write_index(self.small_index, str(self.small_index_path))
        faiss.write_index(self.base_index, str(self.base_index_path))
        payload = {"small": self.small_records, "base": self.base_records}
        self.meta_path.write_text(json.dumps(payload, indent=2))

    def load(self) -> None:
        faiss = _faiss()
        if self.meta_path.exists():
            data = json.loads(self.meta_path.read_text())
            if isinstance(data, dict):
                self.small_records = list(data.get("small", []))
                self.base_records = list(data.get("base", []))
            else:
                self.small_records = []
                self.base_records = []
        if self.small_index_path.exists() and len(self.small_records) == faiss.read_index(str(self.small_index_path)).ntotal:
            self.small_index = faiss.read_index(str(self.small_index_path))
        else:
            self.small_index = faiss.IndexFlatIP(SMALL_DIM)
            self.small_records = []
        if self.base_index_path.exists() and len(self.base_records) == faiss.read_index(str(self.base_index_path)).ntotal:
            self.base_index = faiss.read_index(str(self.base_index_path))
        else:
            self.base_index = faiss.IndexFlatIP(BASE_DIM)
            self.base_records = []
        total = len(self.small_records) + len(self.base_records)
        if total:
            print(f"[store] Loaded {len(self.small_records)} small + {len(self.base_records)} base")
        else:
            print("[store] Fresh index")


def get_store() -> VectorStore:
    return VectorStore()
