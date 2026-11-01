#!/usr/bin/env python3
"""Enroll one or more photos of an object. Labels are for display and voice match."""

from __future__ import annotations

import argparse
import uuid
from datetime import datetime

import numpy as np

from embedding_engine import embed_image, embed_with_fallback
from vector_store import VectorStore


def transcribe_label(seconds: float = 3.0, prompt: str = "say the object name") -> str:
    """Record the microphone and transcribe with Whisper tiny."""
    import sounddevice as sd
    import whisper

    sample_rate = 16000
    print(f"[voice] Recording {seconds:.0f}s — {prompt}...")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("[voice] Transcribing...")
    model = whisper.load_model("tiny")
    text = model.transcribe(np.asarray(audio, dtype=np.float32).reshape(-1), fp16=False)["text"]
    label = " ".join(text.strip().split())
    if not label:
        raise RuntimeError("Whisper returned an empty label")
    print(f"[voice] Heard: {label}")
    return label


def _store_embedding(store: VectorStore, embedding: np.ndarray, metadata: dict, yolo_found: bool) -> None:
    store.add(id=str(uuid.uuid4()), embedding=embedding, metadata=metadata)
    found = "yes" if yolo_found else "no"
    print(f"Enrolled {metadata['label']} with {embedding.shape[0]} dim | YOLO found: {found}")


def enroll(
    image_path: str,
    user_label: str,
    location: str = "unknown",
    tap_x: float | None = None,
    tap_y: float | None = None,
    store: VectorStore | None = None,
) -> VectorStore:
    """Embed one photo with small, and with base when this is a hard case."""
    emb_small, crop, yolo_found = embed_image(image_path, tap_x=tap_x, tap_y=tap_y)
    emb_base = embed_with_fallback(crop, small_score_hint=0.0)
    # YOLO/ONNX before FAISS. Opening the store earlier segfaults on this Mac.
    if store is None:
        store = VectorStore()
    metadata = {
        "label": user_label,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": location,
        "image_path": image_path,
    }
    _store_embedding(store, emb_small, metadata, yolo_found)
    if emb_base.shape[0] != emb_small.shape[0]:
        _store_embedding(store, emb_base, metadata, yolo_found)
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll an object by visual embedding")
    parser.add_argument("--image", default=None, help="Path to one object photo")
    parser.add_argument(
        "--images",
        nargs="+",
        default=None,
        help="Several photos of the same object",
    )
    parser.add_argument("--label", default=None, help="Human-readable name (display only)")
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Record 3 seconds from the mic and use Whisper tiny as the label",
    )
    parser.add_argument("--location", default="unknown", help="Optional location tag")
    args = parser.parse_args()

    images: list[str] = []
    if args.image:
        images.append(args.image)
    if args.images:
        images.extend(args.images)
    if not images:
        parser.error("Provide --image or --images")
    if args.voice:
        label = transcribe_label(3.0)
    elif args.label:
        label = args.label
    else:
        parser.error("Provide --label or --voice")

    store: VectorStore | None = None
    for image_path in images:
        store = enroll(image_path, label, location=args.location, store=store)


if __name__ == "__main__":
    main()
