# RecallOS - One centroid memory for VR glasses

**v0.1.1 Live: Anti-overfit + format-agnostic**

Before: unrelated 0.54 (saved 7 floral background crops)
After: unrelated 0.11, gap 0.19 -> 0.37

Kept 4 / skipped 7 no-bottle, variance 0.0030, 1 centroid
pink_chair.jpeg -> 0.48
bottle1.jpg -> 0.25
unrelated.jpg -> 0.11

Threshold 0.35, DINOv2-S 384-dim, YOLO crop, FAISS small+base
Run: cd mac-prototype && python3 recall.py --image data/pink_chair.jpeg
