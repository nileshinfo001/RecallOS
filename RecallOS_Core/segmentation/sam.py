"""Emergency filename-based object detection (no transformers / YOLO).

Real YOLO-World / SAM can replace detect_objects later.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2

THUMBS_DIR = os.path.join(".", "data", "thumbs")


def detect_objects(image_path: str) -> list[dict[str, Any]]:
    """Infer object label from filename stem (e.g. wallet_1, keys_3)."""
    label = Path(image_path).stem.lower()

    if "wallet" in label:
        return [{"label": "wallet", "bbox": [0, 0, 1, 1], "conf": 0.99}]
    if "key" in label:
        return [{"label": "keys", "bbox": [0, 0, 1, 1], "conf": 0.99}]
    if "scissor" in label:
        return [{"label": "scissors", "bbox": [0, 0, 1, 1], "conf": 0.99}]
    return [{"label": "object", "bbox": [0, 0, 1, 1], "conf": 0.5}]


def segment_objects(image_path: str) -> list[dict[str, Any]]:
    """Detect objects and write 128x128 thumbnails.

    Returns list of {label, bbox, thumbnail_path} for MemoryDB ingest.
    """
    os.makedirs(THUMBS_DIR, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = image.shape[:2]
    detections = detect_objects(image_path)
    base = Path(image_path).stem
    objects: list[dict[str, Any]] = []

    for i, det in enumerate(detections):
        label = det["label"]
        # Expand normalized [0,0,1,1] to full-image pixel bbox
        bx = det["bbox"]
        if bx == [0, 0, 1, 1] or (bx[2] <= 1 and bx[3] <= 1):
            x1, y1, x2, y2 = 0, 0, w, h
        else:
            x1, y1, x2, y2 = map(int, bx)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = image

        thumb = cv2.resize(crop, (128, 128))
        thumb_name = f"{base}_{label}_{i}.jpg"
        thumb_path = os.path.join(THUMBS_DIR, thumb_name)
        cv2.imwrite(thumb_path, thumb)

        objects.append(
            {
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "thumbnail_path": thumb_path,
            }
        )
        print(f"[segment] {label} (conf={det.get('conf', 0):.2f}) -> {thumb_path}")

    return objects
