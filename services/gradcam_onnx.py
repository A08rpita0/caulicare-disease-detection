"""Grad-CAM without TensorFlow, from the ONNX model's activations.

Keras Grad-CAM (services/predictor.py, `_forward`) differentiates the top class score with respect
to the last convolution's output. For this network that gradient has a closed form, because
everything after that convolution is  MaxPool(2) -> Flatten -> Dense(ReLU) -> Dense:

    score        = hidden . W_out[:, top]                    (pre-softmax logit of the top class)
    d score / d hidden_pre = W_out[:, top] * (hidden > 0)     (ReLU gradient)
    d score / d flat       = W_hidden @ that                  (W_hidden: flat -> hidden kernel)
    d score / d pooled     = reshape of that to (H/2, W/2, C)

Grad-CAM weights are the spatial mean of d score / d conv_out per channel. Max pooling sends each
pooled cell's gradient to one position of its 2 x 2 window, so the *sum* over positions per channel
(and therefore the mean) does not depend on which position is chosen: no arg-max bookkeeping is
needed, and ties between equal activations cannot change the result.

The ONNX graph exposes `conv_features` (the last convolution's output after ReLU, NHWC) and
`hidden` (the Dense(ReLU) output); `models/<name>.gradcam.npz`, written by
services/onnx_export.py, holds W_hidden, W_out and the pooling size.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SIDECAR_KEYS = ("w_hidden", "w_out", "pool", "layer", "source_sha256")


def sidecar_path(onnx_path: Path) -> Path:
    return onnx_path.with_name(onnx_path.stem + ".gradcam.npz")


def load_sidecar(path: Path) -> dict:
    """Read the Grad-CAM parameters (numpy only, no pickle)."""
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in SIDECAR_KEYS if k not in z.files]
        if missing:
            raise ValueError(f"{path.name} lacks {missing}")
        return {"w_hidden": z["w_hidden"].astype(np.float32), "w_out": z["w_out"].astype(np.float32),
                "pool": int(z["pool"]), "layer": str(z["layer"]), "source_sha256": str(z["source_sha256"])}


def cam_from_features(conv: np.ndarray, hidden: np.ndarray, top: int, w_hidden: np.ndarray,
                      w_out: np.ndarray, pool: int) -> np.ndarray:
    """conv: (H, W, C) last-convolution activations; hidden: (D,) Dense(ReLU) activations.
    Returns the (H, W) heat-map scaled to [0, 1] (all zeros when nothing contributes)."""
    h, w, c = conv.shape
    g_hidden = w_out[:, top] * (hidden > 0)
    g_pooled = (w_hidden @ g_hidden).reshape(h // pool, w // pool, c)
    weights = g_pooled.sum(axis=(0, 1)) / (h * w)
    cam = np.maximum((conv * weights).sum(axis=-1), 0.0)
    peak = cam.max()
    return cam / peak if peak > 0 else cam
