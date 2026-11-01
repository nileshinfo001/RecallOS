#!/usr/bin/env python3
"""Recall by visual similarity. Falls back to DINOv2-base when small is unsure."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from embedding_engine import embed_image, embed_with_fallback
from enroll import transcribe_label
from vector_store import VectorStore

_STOP = {
    "find", "my", "the", "a", "an", "where", "is", "are", "me", "please",
    "of", "to", "for", "you", "can",
}


def _filter_tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [word for word in words if word not in _STOP]


def _merge(hits: list[dict]) -> list[dict]:
    """Keep the higher score when the same photo is in both indexes."""
    best: dict[tuple, dict] = {}
    for hit in hits:
        key = (hit.get("label"), Path(str(hit.get("image_path", ""))).stem.lower())
        if key not in best or hit["score"] > best[key]["score"]:
            best[key] = hit
    return sorted(best.values(), key=lambda item: item["score"], reverse=True)


def _print_best(hit: dict, threshold: float) -> None:
    voice = "yes" if hit.get("voice_match") else "no"
    print(
        f"Found {hit.get('label', 'unknown')} | Similarity: {hit['score']:.2f} | "
        f"Dim: {hit.get('dim', '?')} | Voice match: {voice}"
    )
    if hit["score"] < threshold:
        print(f"[recall] Below threshold {threshold:.2f}, showing it anyway")


def recall(
    query_image_path: str,
    k: int = 3,
    threshold: float = 0.65,
    text_filter: str = "",
    top: int | None = None,
) -> list[dict]:
    emb_small, crop, _yolo_found = embed_image(query_image_path)
    store = VectorStore()
    depth = max(k, top or 0, 10)
    small_hits = store.search(emb_small, k=depth, apply_threshold=False)
    best_small = max((hit["score"] for hit in small_hits), default=0.0)
    print(f"[recall] Small best similarity: {best_small:.2f}")

    hits = list(small_hits)
    if best_small < 0.55:
        emb_base = embed_with_fallback(crop, small_score_hint=best_small)
        if emb_base.shape[0] != emb_small.shape[0]:
            hits.extend(store.search(emb_base, k=depth, apply_threshold=False))

    ranked = _merge(hits)
    tokens = _filter_tokens(text_filter)
    for hit in ranked:
        label = str(hit.get("label", "")).lower()
        matched = bool(tokens) and any(token in label for token in tokens)
        hit["voice_match"] = matched
        if matched:
            hit["score"] = min(1.0, float(hit["score"]) * 1.2)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    limit = top if top else 1
    chosen = ranked[:limit]

    if not chosen:
        print("No memories stored")
        return []

    if top:
        for rank, hit in enumerate(chosen, start=1):
            print(f"{rank}. {hit.get('label', 'unknown')} | Similarity: {hit['score']:.2f}")
    else:
        _print_best(chosen[0], threshold)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall an object from a query photo")
    parser.add_argument("--image", required=True, help="Query photo")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--top", type=int, default=None, help="Print this many labels, highest score first")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--label", default="", help="Text filter, for example 'pink bottle'")
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Record 3 seconds and use the transcription as the text filter",
    )
    args = parser.parse_args()
    text_filter = transcribe_label(3.0, prompt="say what to find") if args.voice else args.label
    recall(args.image, k=args.k, threshold=args.threshold, text_filter=text_filter, top=args.top)


if __name__ == "__main__":
    main()
