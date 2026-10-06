"""Test fixtures.

The real trained weights are not in the repo, so pipeline tests use an
*untrained* network with the exact architecture from notebooks/Model.ipynb,
saved to and re-loaded from a temporary .h5 file. This verifies loading,
preprocessing, output mapping and Grad-CAM end to end. It says nothing about
accuracy and is never used by the application itself.
"""
import io
import os

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from app import create_app  # noqa: E402
from config import TestConfig  # noqa: E402
from services.predictor import CLASS_NAMES, Predictor  # noqa: E402
from services.similarity_client import SimilarityClient  # noqa: E402


class FakeLeafCheck(SimilarityClient):
    """Stands in for the similarity service in app tests.

    verdict=True/False: the cauliflower-leaf check answers accept/reject (score 0.9 / 0.1,
    threshold 0.5); verdict=None: the check could not be made. Similarity search reports
    "not set up". Calls are counted so tests can assert what ran."""

    def __init__(self, verdict=True):
        super().__init__("")
        self.verdict = verdict
        self.verify_calls = 0
        self.search_calls = 0

    def verify(self, data, filename="image", mimetype="application/octet-stream"):
        self.verify_calls += 1
        if self.verdict is None:
            return {"checked": False}
        return {"checked": True, "is_cauliflower_leaf": self.verdict, "score": 0.9 if self.verdict else 0.1,
                "threshold": 0.5}

    def search(self, data, filename="image", mimetype="application/octet-stream"):
        self.search_calls += 1
        return super().search(data, filename, mimetype)


def build_notebook_architecture():
    import tensorflow as tf
    from tensorflow.keras import layers, models

    size = 256
    resize_and_rescale = tf.keras.Sequential([layers.Resizing(size, size), layers.Rescaling(1.0 / 255)])
    augmentation = tf.keras.Sequential([layers.RandomFlip("horizontal_and_vertical"), layers.RandomRotation(0.2)])
    model = models.Sequential(
        [resize_and_rescale, augmentation, layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(2)]
        + sum([[layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(2)] for _ in range(5)], [])
        + [layers.Flatten(), layers.Dense(64, activation="relu"), layers.Dense(len(CLASS_NAMES), activation="softmax")]
    )
    model.build(input_shape=(None, size, size, 3))
    return model


@pytest.fixture(scope="session")
def model_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "model1.h5"
    build_notebook_architecture().save(path)
    return path


@pytest.fixture(scope="session")
def loaded_predictor(model_file):
    p = Predictor(model_file)
    p.load()
    return p


@pytest.fixture
def make_client():
    """Test client whose cauliflower-leaf check accepts by default (leaf=True)."""
    def _make(predictor=None, config=TestConfig, leaf=True, similarity=None):
        app = create_app(config, predictor=predictor or Predictor("does/not/exist.h5"),
                         similarity=similarity or FakeLeafCheck(leaf))
        return app.test_client()

    return _make


def image_bytes(fmt="PNG", size=(300, 300), color=None, noise=True):
    rng = np.random.default_rng(0)
    if noise:
        arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    else:
        arr = np.full((size[1], size[0], 3), color if color is not None else 120, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


def leafy_bytes(fmt="JPEG", size=(300, 300)):
    """Plant-coloured textured image that passes the leaf-likeness gate (not a real leaf)."""
    rng = np.random.default_rng(1)
    h, w = size[1], size[0]
    arr = np.stack([60 + rng.normal(0, 12, (h, w)), 140 + rng.normal(0, 12, (h, w)), 70 + rng.normal(0, 12, (h, w))], -1)
    buf = io.BytesIO()
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(buf, format=fmt)
    return buf.getvalue()


def flat_bytes(value, size=(300, 300), fmt="PNG"):
    """A uniform image of one grey level (blank / too dark / overexposed frames)."""
    return image_bytes(fmt=fmt, size=size, color=value, noise=False)
