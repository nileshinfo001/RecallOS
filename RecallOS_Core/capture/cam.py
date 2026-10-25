"""Camera capture module for RecallOS."""

import os
from datetime import datetime

import cv2


DATA_DIR = os.path.join(".", "data")


def capture_image(location: str = "bedroom", camera_index: int = 0) -> str:
    """Capture a frame from the webcam and save it under ./data/.

    Filename format: {location}_{DD-MM-YYYY}_{HH-MM-SS}.jpg
    Returns the saved image path.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Failed to capture frame from camera")

    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H-%M-%S")
    filename = f"{location}_{date_str}_{time_str}.jpg"
    path = os.path.join(DATA_DIR, filename)

    cv2.imwrite(path, frame)
    print(f"[capture] Saved: {path}")
    return path


def capture_from_file(source_path: str, location: str = "bedroom") -> str:
    """Copy / re-save an existing image into ./data/ with a timestamped name.

    Keeps the source stem in the filename so emergency label detection
    (wallet_1, keys_3, …) still works after re-save.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    frame = cv2.imread(source_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {source_path}")

    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H-%M-%S")
    ms = now.strftime("%f")[:3]
    # Preserve original stem (e.g. keys_3) for filename-based detection
    stem = os.path.splitext(os.path.basename(source_path))[0]
    filename = f"{location}_{stem}_{date_str}_{time_str}-{ms}.jpg"
    path = os.path.join(DATA_DIR, filename)

    cv2.imwrite(path, frame)
    print(f"[capture] Saved from file: {path}")
    return path
