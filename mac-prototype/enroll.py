#!/usr/bin/env python3
"""Enroll an object from an image using DINOv2 visual embeddings."""

from __future__ import annotations

import argparse
import uuid
from datetime import datetime

from embedding_engine import get_embedding
from vector_store import VectorStore


def enroll(image_path: str, user_label: str, location: str = "unknown") -> None:
    embedding = get_embedding(image_path)
    metadata = {
        "label": user_label,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": location,
        "image_path": image_path,
    }
    store = VectorStore()
    store.add(id=str(uuid.uuid4()), embedding=embedding, metadata=metadata)
    print(f"Enrolled {user_label} with embedding shape {embedding.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll an object by visual embedding")
    parser.add_argument("--image", required=True, help="Path to object photo")
    parser.add_argument("--label", required=True, help="Human-readable name (display only)")
    parser.add_argument("--location", default="unknown", help="Optional location tag")
    args = parser.parse_args()
    enroll(args.image, args.label, args.location)


if __name__ == "__main__":
    main()
