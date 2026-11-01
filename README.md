# RecallOS - One centroid memory for VR glasses

## Problem
VR glasses capture 11 frames of your bottle but save all 11, including 7 floral background crops. Result: unrelated image matches at 0.54, gap only 0.19.

## Solution
One 384-dim centroid per object, not 11 frames.

- YOLO bottle detection (conf <0.01 = skip) + center 35% fallback
- DINOv2-S ONNX (dinov2-small.onnx 84MB) embeddings
- 1 centroid per label: variance 0.0030 check
- FAISS small (hot) + base (cold) index, threshold 0.35
- ONNX ready for Quest 3 passthrough (quest-build/ pending)

## Verified v0.2 (Mac prototype)
Before: unrelated 0.54 (saved 7 background crops)
After: unrelated 0.11, gap 0.19 -> 0.37

Terminal proof:
[store] Loaded 1 small + 0 base
Kept 4 / skipped 7 (no bottle: 7)
Centroid variance: 0.0030
Enrolled blue_bottle with 384 dim | source: 1 centroid, not 11 frames

pink_chair.jpeg -> 0.48
    1. blue_bottle | 0.48
    2. pink_bottle | 0.48
unrelated.jpg -> 0.11
    1. blue_bottle | 0.11

Commands:
python3 enroll_video.py --video data/bottlevideo.mp4 --label pink_bottle
python3 recall.py --image data/pink_chair.jpeg --top 3 --threshold 0.4
python3 enroll_video.py --forget pink_bottle

## v0.3 Roadmap
Quest 3 build: passthrough enroll/recall/forget gestures
30sec video: enroll -> recall 0.48 vs 0.11 -> forget

Run: cd mac-prototype && python3 recall.py --image data/pink_chair.jpeg --top 3
