"""Checks against the real trained models/model1.h5 and the original dataset.

Skipped unless the dataset is available: either in data/raw/Original Dataset or
in the folder named by the CAULICARE_DATASET_DIR environment variable.
"""
import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from config import BASE_DIR
from services.image_validation import assess_quality
from services.predictor import CLASS_NAMES, Predictor

MODEL = BASE_DIR / "models" / "model1.h5"
SPLITS = BASE_DIR / "data" / "splits.json"
DATA = Path(os.getenv("CAULICARE_DATASET_DIR") or BASE_DIR / "data" / "raw" / "Original Dataset")

pytestmark = pytest.mark.skipif(not (MODEL.exists() and SPLITS.exists() and DATA.exists()),
                                reason="real model/dataset not present")


def _split(name):
    return json.loads(SPLITS.read_text())["splits"][name]


@pytest.fixture(scope="module")
def real_predictor():
    p = Predictor(MODEL)
    p.load()
    return p


def _post(client, rel, explain=True):
    path = DATA / rel
    return client.post(f"/api/predict{'' if explain else '?explain=0'}",
                       data={"image": (io.BytesIO(path.read_bytes()), Path(rel).name)},
                       content_type="multipart/form-data")


def test_real_model_contract(real_predictor):
    m = real_predictor.load()
    assert m.input_shape[1:] == (256, 256, 3) and m.output_shape[-1] == 4
    assert m.count_params() == 183_812


def test_served_pipeline_reproduces_reported_test_accuracy(real_predictor, make_client):
    """All 99 held-out test images through the real API: 87 correct = 87.9%, as documented."""
    client = make_client(predictor=real_predictor)
    test = _split("test")
    correct = 0
    for e in test:
        r = _post(client, e["path"], explain=False)
        assert r.status_code == 200 and r.json["status"] == "ok", (e["path"], r.json.get("status"))
        correct += r.json["prediction"]["disease"] == CLASS_NAMES[e["label"]]
    assert len(test) == 99 and correct == 87


def test_gradcam_on_real_images(real_predictor, make_client):
    client = make_client(predictor=real_predictor)
    for c in range(len(CLASS_NAMES)):
        e = next(x for x in _split("test") if x["label"] == c)
        p = _post(client, e["path"]).json["prediction"]
        assert p["explanation"]["overlay"].startswith("data:image/jpeg;base64,")


def test_no_dataset_photo_is_refused_by_the_gates():
    for name in ("train", "val", "test"):
        for e in _split(name):
            img = Image.open(DATA / e["path"])
            img.load()
            q = assess_quality(img.convert("RGB"))
            assert q.acceptable and q.looks_like_leaf, (e["path"], q.to_dict())
