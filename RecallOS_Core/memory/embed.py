"""Text embedding via sentence-transformers (all-MiniLM-L6-v2 → 384-dim)."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("[embed] Loading all-MiniLM-L6-v2...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_text(text: str) -> np.ndarray:
    """Embed a text string into a 384-dimensional L2-normalized float32 vector."""
    model = _get_model()
    vec = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32)


def embed_label(label: str) -> np.ndarray:
    """Embed an object label (alias for embed_text for clarity)."""
    return embed_text(label)
