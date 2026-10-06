"""Cauliflower-leaf validator: decides whether an upload should reach the disease model.

The disease CNN always answers with one of its four classes, whatever the
image shows, so its output cannot tell a cauliflower leaf from a cat. This
check runs first and uses a different model: a logistic-regression head on
the same DINOv2-small embedding that similarity search already computes.

  embedding (384-d, L2-normalised) -> w . x + b -> sigmoid -> score in [0, 1]
  score >= threshold  -> cauliflower leaf, continue to disease analysis
  score <  threshold  -> rejected, the disease model is never run

Weights, bias and threshold are read from index/leaf_validator.json, which is
produced by scripts/train_leaf_validator.py together with the measured
metrics in reports/leaf_validator.json. The threshold was chosen on a
validation split and checked on a separate test split; it is not hand-tuned.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

EMBEDDING_DIM = 384


class ValidatorUnavailableError(RuntimeError):
    """The validator file is missing or malformed."""


class LeafValidator:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.weights: np.ndarray | None = None
        self.bias = 0.0
        self.threshold = 0.5
        self.info: dict = {}

    @property
    def loaded(self) -> bool:
        return self.weights is not None

    def load(self) -> "LeafValidator":
        try:
            spec = json.loads(self.path.read_text(encoding="utf-8"))
            w = np.asarray(spec["weights"], dtype=np.float32)
            bias, threshold = float(spec["bias"]), float(spec["threshold"])
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValidatorUnavailableError(f"cannot read {self.path.name}") from exc
        if w.shape != (EMBEDDING_DIM,) or not np.all(np.isfinite(w)) or not 0.0 < threshold < 1.0 \
                or not math.isfinite(bias):
            raise ValidatorUnavailableError("invalid validator parameters")
        self.weights, self.bias, self.threshold = w, bias, threshold
        self.info = {k: spec[k] for k in ("model", "version") if isinstance(spec.get(k), str)}
        return self

    def score(self, embedding: np.ndarray) -> float:
        if self.weights is None:
            raise ValidatorUnavailableError("validator not loaded")
        z = float(np.dot(self.weights, embedding.astype(np.float32))) + self.bias
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))

    def check(self, embedding: np.ndarray) -> dict:
        s = self.score(embedding)
        return {"is_cauliflower_leaf": s >= self.threshold, "score": round(s, 4), "threshold": round(self.threshold, 4)}
