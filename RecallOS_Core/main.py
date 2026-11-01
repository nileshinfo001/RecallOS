#!/usr/bin/env python3
"""RecallOS demo — capture → segment → embed → store → query."""

from __future__ import annotations

import glob
import os
import sys
from datetime import datetime

# Ensure package-relative imports work when run as `python main.py`
sys.path.insert(0, ".")

from capture.cam import capture_from_file, capture_image
from memory.embed import embed_label
from memory.faiss_db import MemoryDB
from query.search import search_memory
from segmentation.sam import segment_objects

SAMPLES_DIR = os.path.join(".", "data", "samples")
DATA_DIR = os.path.join(".", "data")
THUMBS_DIR = os.path.join(".", "data", "thumbs")
LOCATION = "bedroom"
NUM_CAPTURES = 5


def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(THUMBS_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _is_image(path: str) -> bool:
    return path.lower().endswith(_IMAGE_EXTS)


def gather_images() -> list[str]:
    """Prefer sample images; fall back to live camera captures."""
    samples = sorted(
        path for path in glob.glob(os.path.join(SAMPLES_DIR, "*")) if _is_image(path)
    )

    paths: list[str] = []

    if samples:
        print(f"[main] Found {len(samples)} sample image(s) — using demo mode")
        for src in samples[:NUM_CAPTURES]:
            paths.append(capture_from_file(src, location=LOCATION))
        # If fewer than NUM_CAPTURES samples, reuse them
        while len(paths) < NUM_CAPTURES and samples:
            src = samples[len(paths) % len(samples)]
            paths.append(capture_from_file(src, location=LOCATION))
    else:
        print("[main] No samples found — capturing from camera")
        for i in range(NUM_CAPTURES):
            print(f"[main] Capture {i + 1}/{NUM_CAPTURES} — look at the camera...")
            try:
                paths.append(capture_image(location=LOCATION))
            except RuntimeError as e:
                print(f"[main] Camera error: {e}")
                break

    return paths


def ingest(paths: list[str], db: MemoryDB) -> None:
    """Segment each image, embed labels, add to MemoryDB."""
    for path in paths:
        print(f"\n[main] Processing {path}")
        objects = segment_objects(path)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for obj in objects:
            emb = embed_label(obj["label"])
            db.add(
                {
                    "object_name": obj["label"],
                    "timestamp": ts,
                    "location": LOCATION,
                    "embedding": emb,
                    "thumbnail_path": obj["thumbnail_path"],
                }
            )

    db.save()


def main() -> None:
    print("=" * 50)
    print("  RecallOS — AI memory for the physical world")
    print("=" * 50)

    ensure_dirs()
    db = MemoryDB()

    paths = gather_images()
    if not paths:
        print(
            "[main] Nothing to process.\n"
            "  → Drop .jpg/.png files into data/samples/  OR  connect a webcam."
        )
        return

    ingest(paths, db)

    print("\n" + "-" * 50)
    print("[main] Querying: \"where is my wallet\"")
    print("-" * 50)
    hits = search_memory("where is my wallet", db, top_k=3)

    if not hits:
        # Helpful hint when the demo images don't contain "keys"
        print(
            "\n[main] Tip: Name samples like keys_1.jpg / wallet_2.jpg "
            "so filename-based detection can label them."
        )
        # Also dump what's in the DB so the user can see available labels
        if len(db) > 0:
            labels = sorted({m["object_name"] for m in db.metadata})
            print(f"[main] Objects currently in memory: {labels}")

    print("\n[main] Done.")


if __name__ == "__main__":
    main()
