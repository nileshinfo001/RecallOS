#!/usr/bin/env python3
"""Enroll the object under a click. The click is not used as a search key."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from embedding_engine import load_rgb

from enroll import enroll


def tap_enroll(image_path: str, user_label: str, location: str = "unknown") -> None:
    img = load_rgb(image_path)
    width, height = img.size

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Click the object to enroll")
    ax.axis("off")
    print("[tap] Click the object in the image window")
    clicks = plt.ginput(1, timeout=-1)
    plt.close(fig)
    if not clicks:
        raise RuntimeError("No click received")

    x, y = clicks[0]
    tap_x = min(1.0, max(0.0, float(x) / width))
    tap_y = min(1.0, max(0.0, float(y) / height))
    print(f"[tap] Click ({x:.0f}, {y:.0f}) -> ({tap_x:.3f}, {tap_y:.3f})")
    enroll(image_path, user_label, location=location, tap_x=tap_x, tap_y=tap_y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll by clicking an object")
    parser.add_argument("--image", required=True, help="Path to object photo")
    parser.add_argument("--label", required=True, help="Human-readable name (display only)")
    parser.add_argument("--location", default="unknown", help="Optional location tag")
    args = parser.parse_args()
    tap_enroll(args.image, args.label, args.location)


if __name__ == "__main__":
    main()
