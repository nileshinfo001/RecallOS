"""DINOv2-small visual embeddings via ONNX Runtime (384-dim, L2-normalized)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ONNX_PATH = Path(__file__).resolve().parent / "dinov2-small.onnx"
HF_MODEL_ID = "facebook/dinov2-small"
INPUT_SIZE = 224
EMBED_DIM = 384
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_session: ort.InferenceSession | None = None
_input_name: str | None = None


def _export_onnx(dest: Path) -> None:
    """Create dinov2-small.onnx from HuggingFace weights if missing."""
    import torch
    from transformers import Dinov2Model

    print(f"[embed] Exporting {HF_MODEL_ID} -> {dest.name} ...")
    model = Dinov2Model.from_pretrained(HF_MODEL_ID)
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
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
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


def preprocess(image_path: str) -> np.ndarray:
    """PIL load → 224x224 RGB → ImageNet normalize → NCHW float32."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, 0).astype(np.float32)


def get_embedding(image_path: str) -> np.ndarray:
    """Return L2-normalized DINOv2 CLS embedding, shape (384,), dtype float32."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    pixel_values = preprocess(image_path)
    session, input_name = _get_session()
    hidden = session.run(None, {input_name: pixel_values})[0]
    # CLS token (index 0). Fallback: mean-pool all tokens.
    if hidden.ndim == 3 and hidden.shape[-1] == EMBED_DIM:
        vector = hidden[0, 0, :]
    else:
        vector = np.mean(hidden.reshape(-1, hidden.shape[-1]), axis=0)

    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.shape[0] != EMBED_DIM:
        raise RuntimeError(f"Expected {EMBED_DIM}-dim embedding, got {vector.shape}")

    norm = np.linalg.norm(vector)
    if norm == 0:
        raise RuntimeError("Zero-norm embedding")
    vector = vector / norm
    return vector.astype(np.float32)
