"""DINOv2-small visual embeddings via ONNX Runtime (384-dim, L2-normalized)."""

from __future__ import annotations

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

_ROOT = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = Path(os.path.join(_ROOT, "dinov2-small.onnx"))
BASE_ONNX_PATH = os.path.join(_ROOT, "dinov2-base.onnx")
YOLO_WEIGHTS = Path(os.path.join(_ROOT, "yolov8n.pt"))
HF_MODEL_ID = "facebook/dinov2-small"
HF_BASE_ID = "facebook/dinov2-base"
INPUT_SIZE = 224
BASE_INPUT_SIZE = 518
EMBED_DIM = 384
BASE_DIM = 768
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def is_image_path(path: str) -> bool:
    """True for still images, ignoring extension case (.JPG and .jpeg both match)."""
    return Path(path).suffix.lower() in IMAGE_EXTS


def image_stem(path: str) -> str:
    """Format-agnostic id: pink_chair.jpeg and pink_chair.JPG are both pink_chair."""
    return Path(path).stem.lower()


def load_rgb(path: str) -> Image.Image:
    """Open jpg, jpeg, png, webp, or bmp. Extension case does not matter."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    if not is_image_path(path):
        raise ValueError(f"Unsupported image format: {path}")
    with Image.open(path) as img:
        return img.convert("RGB")

_session: ort.InferenceSession | None = None
_input_name: str | None = None
_base_session: ort.InferenceSession | None = None
_base_input_name: str | None = None


def _export_onnx(dest: Path, model_id: str = HF_MODEL_ID, size: int = INPUT_SIZE) -> None:
    """Create a DINOv2 ONNX file from HuggingFace weights if missing."""
    import torch
    from transformers import Dinov2Model

    print(f"[embed] Exporting {model_id} -> {dest.name} ({size}px) ...")
    model = Dinov2Model.from_pretrained(model_id)
    model.eval()

    class _DinoONNX(torch.nn.Module):
        def __init__(self, inner: Dinov2Model):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            # last_hidden_state: (B, 1+patches, 384)
            return self.inner(pixel_values=pixel_values).last_hidden_state

    wrapper = _DinoONNX(model)
    wrapper.eval()
    dummy = torch.randn(1, 3, size, size)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Torch 2.14 uses the dynamo exporter (writes dest + dest.data).
    try:
        torch.onnx.export(
            wrapper,
            dummy,
            str(dest),
            input_names=["pixel_values"],
            output_names=["last_hidden_state"],
            dynamo=True,
            opset_version=18,
        )
    except TypeError:
        torch.onnx.export(
            wrapper,
            dummy,
            str(dest),
            input_names=["pixel_values"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "last_hidden_state": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"[embed] Wrote {dest}")


def _get_session() -> tuple[ort.InferenceSession, str]:
    global _session, _input_name
    if _session is None:
        if not ONNX_PATH.exists():
            _export_onnx(ONNX_PATH)
        _session = ort.InferenceSession(
            str(ONNX_PATH),
            providers=["CPUExecutionProvider"],
        )
        _input_name = _session.get_inputs()[0].name
        print(f"[embed] ONNX ready: {ONNX_PATH.name}")
    assert _input_name is not None
    return _session, _input_name


def _clamp_box(w: int, h: int, x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int]:
    x1_i = max(0, min(w - 1, int(round(x1))))
    y1_i = max(0, min(h - 1, int(round(y1))))
    x2_i = max(x1_i + 1, min(w, int(round(x2))))
    y2_i = max(y1_i + 1, min(h, int(round(y2))))
    return x1_i, y1_i, x2_i, y2_i


# 35% keeps the object and drops most of the bedsheet or hand around it.
CENTER_CROP_FRACTION = 0.35


def _center_crop(img: Image.Image, fraction: float = CENTER_CROP_FRACTION) -> Image.Image:
    """Fallback: keep the center of the frame (default 35%)."""
    w, h = img.size
    margin = (1.0 - fraction) / 2.0
    box = _clamp_box(w, h, w * margin, h * margin, w * (1.0 - margin), h * (1.0 - margin))
    print(f"[embed] No bottle — center {fraction:.0%} crop {box}")
    return img.crop(box)


def center_60_crop(image_path: str) -> Image.Image:
    """Center crop used when YOLO cannot run. The window is 35%, not 60%."""
    return _center_crop(load_rgb(image_path))


def _tap_crop(img: Image.Image, tap_x: float, tap_y: float) -> Image.Image:
    """Crop a 40% window centered on a 0–1 tap."""
    w, h = img.size
    cx, cy = tap_x * w, tap_y * h
    box = _clamp_box(w, h, cx - 0.2 * w, cy - 0.2 * h, cx + 0.2 * w, cy + 0.2 * h)
    print(f"[embed] Tap crop @ ({tap_x:.2f}, {tap_y:.2f}) -> {box}")
    return img.crop(box)


def _letterbox(img: Image.Image) -> Image.Image:
    """Pad to a square so the 224 resize does not stretch the object."""
    img = img.convert("RGB")
    w, h = img.size
    side = max(w, h)
    if w == h:
        return img
    canvas = Image.new("RGB", (side, side), (128, 128, 128))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def _bottle_crop(img: Image.Image, image_path: str) -> tuple[Image.Image, bool]:
    """Crop to the highest-confidence YOLOv8n bottle, else center 60%.

    conf=0.01 so a partly hidden bottle still counts. The best bottle box wins.
    A torch/ultralytics failure falls back to the center crop instead of crashing.
    """
    try:
        from ultralytics import YOLO

        weights = "yolov8n.pt" if not YOLO_WEIGHTS.exists() else str(YOLO_WEIGHTS)
        model = YOLO(weights)
        # Pixels, not the path, so .jpeg and .jpg take the same route.
        bgr = np.asarray(img.convert("RGB"))[:, :, ::-1]
        results = model(bgr, verbose=False, conf=0.01)
        best = None
        best_conf = 0.0
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                name = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                if name != "bottle" or conf <= best_conf:
                    continue
                best_conf = conf
                best = box.xyxy[0].tolist()
        if best is None:
            return _center_crop(img), False
        x1, y1, x2, y2 = _clamp_box(img.size[0], img.size[1], *best)
        print(f"[embed] YOLO bottle crop conf={best_conf:.2f} -> {(x1, y1, x2, y2)}")
        return img.crop((x1, y1, x2, y2)), True
    except Exception as e:
        print(f"[YOLO failed: {e}, using center 35% crop]")
        return center_60_crop(image_path), False


def prepare_image(
    image_path: str,
    tap_x: float | None = None,
    tap_y: float | None = None,
) -> tuple[Image.Image, bool]:
    """Return the letterboxed region and whether YOLO found a bottle."""
    img = load_rgb(image_path)
    yolo_found = False
    if tap_x is not None or tap_y is not None:
        if tap_x is None or tap_y is None:
            raise ValueError("tap_x and tap_y must both be set")
        cropped = _tap_crop(img, float(tap_x), float(tap_y))
    else:
        cropped, yolo_found = _bottle_crop(img, image_path)
    return _letterbox(cropped), yolo_found


def preprocess(image: Image.Image | str, size: int = INPUT_SIZE) -> np.ndarray:
    """Square RGB → ImageNet normalize → NCHW float32.

    A path is opened as-is. Pass a cropped PIL image to embed a region.
    """
    img = load_rgb(image) if isinstance(image, str) else image
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, 0).astype(np.float32)


def _l2(vector: np.ndarray, dim: int) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.shape[0] != dim:
        raise RuntimeError(f"Expected {dim}-dim embedding, got {vector.shape}")
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise RuntimeError("Zero-norm embedding")
    return (vector / norm).astype(np.float32)


def _pool(hidden: np.ndarray, pool: str) -> np.ndarray:
    if hidden.ndim != 3:
        return np.mean(hidden.reshape(-1, hidden.shape[-1]), axis=0)
    tokens = hidden[0]
    if pool == "mean" and tokens.shape[0] > 1:
        return tokens[1:].mean(axis=0)
    return tokens[0]


def _run(session: ort.InferenceSession, input_name: str, pixel_values: np.ndarray, pool: str, dim: int) -> np.ndarray:
    hidden = session.run(None, {input_name: pixel_values})[0]
    return _l2(_pool(hidden, pool), dim)


def _get_base_session() -> tuple[ort.InferenceSession, str] | None:
    """Load dinov2-base.onnx when the file exists. Export it on first hard recall."""
    global _base_session, _base_input_name
    if _base_session is None:
        if not os.path.exists(BASE_ONNX_PATH):
            _export_onnx(Path(BASE_ONNX_PATH), model_id=HF_BASE_ID, size=BASE_INPUT_SIZE)
        if not os.path.exists(BASE_ONNX_PATH):
            print("[embed] dinov2-base.onnx missing — staying on small")
            return None
        _base_session = ort.InferenceSession(BASE_ONNX_PATH, providers=["CPUExecutionProvider"])
        _base_input_name = _base_session.get_inputs()[0].name
        print(f"[embed] ONNX ready: {os.path.basename(BASE_ONNX_PATH)}")
    assert _base_input_name is not None
    return _base_session, _base_input_name


def embed_image(
    image_path: str,
    tap_x: float | None = None,
    tap_y: float | None = None,
) -> tuple[np.ndarray, Image.Image, bool]:
    """Crop, letterbox, and embed with DINOv2-small only.

    Returns (emb_small 384-d, letterboxed crop, yolo_found).
    """
    crop, yolo_found = prepare_image(image_path, tap_x=tap_x, tap_y=tap_y)
    session, input_name = _get_session()
    emb_small = _run(session, input_name, preprocess(crop, INPUT_SIZE), "cls", EMBED_DIM)
    return emb_small, crop, yolo_found


def embed_with_fallback(crop: Image.Image, small_score_hint: float = 0.0) -> np.ndarray:
    """Run DINOv2-base on the same crop when the small score is a hard case.

    Mean-pools patch tokens (768-d). Returns the small CLS embedding when base
    is unavailable or the small score is already at least 0.55.
    """
    hard = small_score_hint < 0.55
    if hard:
        loaded = _get_base_session()
        if loaded is not None:
            session, input_name = loaded
            pixels = preprocess(crop, BASE_INPUT_SIZE)
            return _run(session, input_name, pixels, "mean", BASE_DIM)
    session, input_name = _get_session()
    return _run(session, input_name, preprocess(crop, INPUT_SIZE), "cls", EMBED_DIM)


def get_embedding(
    image_path: str,
    tap_x: float | None = None,
    tap_y: float | None = None,
) -> np.ndarray:
    """Return the L2-normalized DINOv2-small CLS embedding, shape (384,)."""
    emb_small, _, _ = embed_image(image_path, tap_x=tap_x, tap_y=tap_y)
    return emb_small
