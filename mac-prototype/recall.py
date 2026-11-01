#!/usr/bin/env python3
"""Recall an object by visual similarity (DINOv2 + FAISS). No filename matching."""

from __future__ import annotations

import argparse

from embedding_engine import get_embedding
from vector_store import VectorStore


def recall(query_image_path: str, k: int = 3, threshold: float = 0.65) -> list[dict]:
    query_embedding = get_embedding(query_image_path)
    store = VectorStore()
    results = store.search(query_embedding, k=k, threshold=threshold)

    if not results:
        print("Not seen recently")
        return []

    for hit in results:
        label = hit.get("label", "unknown")
        score = hit.get("score", 0.0)
        timestamp = hit.get("timestamp", "unknown")
        location = hit.get("location", "unknown")
        print(
            f"Found {label} | Similarity: {score:.2f} | "
            f"Seen: {timestamp} | Location: {location}"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall an object from a query photo")
    parser.add_argument("--image", required=True, help="Query photo (visual search only)")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()
    recall(args.image, k=args.k, threshold=args.threshold)


if __name__ == "__main__":
    main()
