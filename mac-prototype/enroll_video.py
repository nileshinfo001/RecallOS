#!/usr/bin/env python3
"""Enroll a rotating object video as one centroid of a few different views.

Frames that YOLO does not accept, and near-duplicate poses, are not stored.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np

from embedding_engine import embed_image
from vector_store import VectorStore

MAX_KEEP = 5
DUPLICATE_COSINE = 0.92


def extract_frames(video_path: str, out_dir: Path) -> tuple[list[Path], float, int]:
    """Save about 2 frames per second. Step size is native fps / 2."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1:
        fps = 30.0
    step = max(1, int(round(fps / 2.0)))
    saved: list[Path] = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            path = out_dir / f"frame_{len(saved):03d}.jpg"
            cv2.imwrite(str(path), frame)
            saved.append(path)
        index += 1
    cap.release()
    print(f"[video] {fps:.1f} fps, step {step} -> {len(saved)} frames (~2 fps)")
    return saved, fps, step


def _too_similar(embedding: np.ndarray, kept: list[np.ndarray]) -> bool:
    return any(float(np.dot(embedding, previous)) > DUPLICATE_COSINE for previous in kept)


def select_diverse(frame_paths: list[Path]) -> tuple[list[np.ndarray], dict[str, int]]:
    """Keep at most 5 YOLO bottle views that are not the same pose."""
    kept: list[np.ndarray] = []
    counts = {
        "frames": len(frame_paths),
        "kept": 0,
        "skipped_no_bottle": 0,
        "skipped_duplicate": 0,
        "skipped_full": 0,
    }
    for index, path in enumerate(frame_paths):
        if len(kept) >= MAX_KEEP:
            counts["skipped_full"] += len(frame_paths) - index
            break
        embedding, _crop, yolo_found = embed_image(str(path))
        if not yolo_found:
            counts["skipped_no_bottle"] += 1
            print(f"[video] Skip {path.name}: YOLO conf < 0.01")
            continue
        if _too_similar(embedding, kept):
            counts["skipped_duplicate"] += 1
            print(f"[video] Skip {path.name}: cosine > {DUPLICATE_COSINE:.2f} to a kept pose")
            continue
        kept.append(embedding)
        print(f"[video] Keep {path.name} ({len(kept)}/{MAX_KEEP})")
    counts["kept"] = len(kept)
    return kept, counts


def centroid_and_variance(embeddings: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Mean of the kept views, L2-normalized. Variance is of cosine-to-centroid."""
    stacked = np.stack([np.asarray(item, dtype=np.float32) for item in embeddings], axis=0)
    mean = stacked.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0:
        raise RuntimeError("Centroid norm is zero")
    centroid = (mean / norm).astype(np.float32)
    sims = stacked @ centroid
    variance = float(np.var(sims))
    return centroid, variance


def enroll_video(video_path: str, label: str) -> None:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    with tempfile.TemporaryDirectory(prefix="recallos_frames_") as tmp:
        frames, _fps, _step = extract_frames(video_path, Path(tmp))
        if not frames:
            raise RuntimeError("Video produced no frames")
        kept, counts = select_diverse(frames)
    if not kept:
        raise RuntimeError("No bottle frames kept. YOLO did not find a bottle above conf 0.01.")

    centroid, variance = centroid_and_variance(kept)
    # Embeddings are ready, so FAISS loads after YOLO.
    # One centroid per label: a new enroll replaces the previous one.
    store = VectorStore()
    store.drop_label(label)
    store.add(
        id=str(uuid.uuid4()),
        embedding=centroid,
        metadata={
            "label": label,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "location": "video",
            "image_path": video_path,
            "source": "video_centroid",
            "frames_kept": counts["kept"],
            "variance": variance,
        },
    )
    skipped = counts["skipped_no_bottle"] + counts["skipped_duplicate"] + counts["skipped_full"]
    print(
        f"Kept {counts['kept']} / skipped {skipped} "
        f"(no bottle: {counts['skipped_no_bottle']}, duplicate pose: {counts['skipped_duplicate']}, "
        f"over cap: {counts['skipped_full']})"
    )
    print(f"Centroid variance: {variance:.4f}")
    print(f"Enrolled {label} with {centroid.shape[0]} dim | source: 1 centroid, not {counts['frames']} frames")
    print("Test:")
    print("  python3 recall.py --image data/pink_chair.jpeg --threshold 0.4")
    print("  python3 recall.py --image data/unrelated.jpg --threshold 0.4")


def forget_label(label: str) -> None:
    """Remove a label from metadata.json and rebuild faiss_small.index."""
    store = VectorStore()
    removed = store.drop_label(label)
    if removed == 0:
        print(f"No memory labeled {label}")
        return
    print(f"Forgot {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll a bottle video as one diverse centroid")
    parser.add_argument("--video", default=None, help="Path to the object video")
    parser.add_argument("--label", default=None, help="Human-readable name (display only)")
    parser.add_argument("--forget", default=None, help="Remove this label and rebuild the index")
    args = parser.parse_args()
    if args.forget:
        forget_label(args.forget)
        return
    if not args.video or not args.label:
        parser.error("Provide --video and --label, or --forget")
    enroll_video(args.video, args.label)


if __name__ == "__main__":
    main()
