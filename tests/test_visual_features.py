"""Affected-area estimate, and how /api/predict combines it with the similarity service."""
import io

import numpy as np
import pytest
from PIL import Image

from app import create_app
from config import TestConfig
from services.predictor import Predictor
from services.severity import SeverityError, band_for, estimate
from services.similarity_client import NOT_CONFIGURED, SimilarityClient
from tests.conftest import image_bytes, leafy_bytes


def solid(color, size=(64, 64)):
    return Image.new("RGB", size, color)


class FakeSimilarity(SimilarityClient):
    """Client double that returns a fixed, valid response without any network.
    Its cauliflower-leaf check accepts unless verdict=False."""

    def __init__(self, response=None, verdict=True):
        super().__init__("http://similarity.test", labels=["Downy Mildew"])
        self.response = response or {"available": True, "results": [
            {"id": "downy-mildew-001", "label": "Downy Mildew", "similarity": 0.9,
             "thumbnail": "data:image/jpeg;base64,AAAA"}], "timing_ms": {}}
        self.calls = 0
        self.verdict = verdict

    def verify(self, data, filename="image", mimetype=""):
        return {"checked": True, "is_cauliflower_leaf": self.verdict, "score": 0.9 if self.verdict else 0.1,
                "threshold": 0.5}

    def search(self, data, filename="image", mimetype=""):
        self.calls += 1
        return self.response

    def warm_up(self, min_interval=120.0):
        pass


# ----------------------------------------------------------------- severity
def leaf_with_lesions(lesion_fraction):
    """Blue-green leaf on a dark background with a block of tan 'lesion' tissue inside it."""
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    arr[...] = (20, 20, 20)
    arr[40:260, 40:260] = (70, 140, 120)
    side = int(round(220 * np.sqrt(lesion_fraction)))
    if side:
        arr[100:100 + side, 100:100 + side] = (170, 120, 70)
    arr = np.clip(arr.astype(int) + np.random.default_rng(1).integers(-6, 6, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def test_affected_share_tracks_lesion_area():
    healthy = estimate(leaf_with_lesions(0)).affected_percent
    moderate = estimate(leaf_with_lesions(0.2)).affected_percent
    assert healthy < 2
    assert 12 < moderate < 28  # ~20% of the leaf area is lesion-coloured


@pytest.mark.parametrize("pct,band", [(0, "Low"), (9.9, "Low"), (10, "Moderate"), (29.9, "Moderate"),
                                      (30, "High"), (60, "Very high"), (100, "Very high")])
def test_severity_bands(pct, band):
    assert band_for(pct) == band


def test_severity_visualisation_outputs():
    est = estimate(leaf_with_lesions(0.2)).to_dict()
    assert est["available"] and est["band"] in {"Low", "Moderate", "High", "Very high"}
    assert est["mask"].startswith("data:image/png;base64,")
    assert est["overlay"].startswith("data:image/jpeg;base64,")
    assert "not a validated" in est["band_note"]


def test_severity_refuses_images_without_leaf():
    with pytest.raises(SeverityError):
        estimate(solid((200, 30, 30), size=(300, 300)))
    with pytest.raises(SeverityError):
        estimate(solid((0, 0, 0), size=(10, 10)))


def test_severity_refuses_cauliflower_head():
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    arr[...] = (70, 140, 110)
    arr[60:240, 60:240] = (235, 225, 190)  # cream curd filling the centre
    with pytest.raises(SeverityError, match="head"):
        estimate(Image.fromarray(arr))


# ---------------------------------------------------------------------- API
def upload(client, data, name="leaf.jpg"):
    return client.post("/api/predict", data={"image": (io.BytesIO(data), name)}, content_type="multipart/form-data")


def test_predict_response_includes_all_sections(loaded_predictor):
    sim = FakeSimilarity()
    body = upload(create_app(TestConfig, predictor=loaded_predictor, similarity=sim).test_client(), leafy_bytes()).json
    assert body["status"] == "ok"
    for key in ("prediction", "quality", "reasons", "disease", "disclaimer"):  # existing contract
        assert key in body
    assert body["prediction"]["explanation"]["overlay"].startswith("data:image/jpeg;base64,")  # Grad-CAM
    assert "available" in body["severity"]
    assert body["similar_images"]["available"] and body["similar_images"]["results"][0]["label"] == "Downy Mildew"
    assert sim.calls == 1


def test_predict_works_when_similar_images_are_not_set_up(loaded_predictor, make_client):
    # The leaf check passes (FakeLeafCheck) but its similarity search is not configured.
    body = upload(make_client(predictor=loaded_predictor), leafy_bytes()).json
    assert body["status"] == "ok" and body["prediction"]["disease"]
    assert body["similar_images"] == {"available": False, "reason": NOT_CONFIGURED}


@pytest.mark.parametrize("data,verdict,status", [
    (image_bytes(), False, "not_cauliflower_leaf"),            # rejected by the leaf validator
    (image_bytes(color=128, noise=False), True, "low_quality"),  # refused by the quality gate
], ids=["not-a-leaf", "blank"])
def test_refused_images_get_no_similarity_or_severity(loaded_predictor, data, verdict, status):
    sim = FakeSimilarity(verdict=verdict)
    client = create_app(TestConfig, predictor=loaded_predictor, similarity=sim).test_client()
    body = upload(client, data, name="x.png").json
    assert body["status"] == status and "similar_images" not in body and "severity" not in body
    assert sim.calls == 0  # similarity search is never run for refused images


def test_health_reports_similarity_configuration():
    client = create_app(TestConfig, predictor=Predictor("does/not/exist.h5"), similarity=FakeSimilarity()).test_client()
    body = client.get("/api/health").json
    assert body["similarity_search"] is True and "reachable" in body["similarity_service"]
    plain = create_app(TestConfig, predictor=Predictor("does/not/exist.h5")).test_client()
    assert plain.get("/api/health").json["similarity_search"] is False


def test_old_thumbnail_route_is_gone(make_client):
    assert make_client().get("/reference/black-rot-001.jpg").status_code == 404


# ---------------------------------------------- speed-ups must not change the estimate
def _pillow_close(mask, size):
    """The original implementation: Pillow's rank filters."""
    from PIL import ImageFilter

    img = Image.fromarray(mask.astype(np.uint8) * 255, "L").filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size))
    return np.asarray(img) > 127


def _brute_force_extreme(a, size, use_max):
    """Definition of the filter: the max/min over each window, clipped where it leaves the image."""
    r, (h, w) = size // 2, a.shape
    out = np.empty_like(a)
    for y in range(h):
        for x in range(w):
            win = a[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
            out[y, x] = win.max() if use_max else win.min()
    return out


@pytest.mark.parametrize("shape", [(30, 41), (7, 7), (5, 12), (12, 5), (1, 9), (20, 20)])
@pytest.mark.parametrize("size", [3, 7, 15])
@pytest.mark.parametrize("use_max", [True, False])
def test_window_extreme_matches_its_definition(shape, size, use_max):
    from services.severity import _window_extreme

    a = (np.random.default_rng(shape[0] * 31 + size).random(shape) < 0.4).astype(np.uint8) * 255
    assert np.array_equal(_window_extreme(a, size, use_max), _brute_force_extreme(a, size, use_max))


@pytest.mark.parametrize("shape", [(288, 384), (384, 288), (101, 77), (15, 15), (14, 40), (40, 14)])
@pytest.mark.parametrize("size", [7, 15, 21])
@pytest.mark.parametrize("density", [0.05, 0.5, 0.95])
def test_fast_closing_equals_pillows_rank_filters(shape, size, density):
    from services.severity import _close

    mask = np.random.default_rng(shape[0] * 7 + shape[1] + size).random(shape) < density
    fast, reference = _close(mask, size), _pillow_close(mask, size)
    r = size // 2
    # Away from the border every Pillow version agrees; at the border only versions that clip the window do.
    assert np.array_equal(fast[r:-r, r:-r], reference[r:-r, r:-r])
    if _pillow_clips_at_the_border():
        assert np.array_equal(fast, reference)


def _pillow_clips_at_the_border():
    from PIL import ImageFilter

    probe = np.zeros((9, 9), dtype=np.uint8)
    probe[0, 0] = 255
    out = np.asarray(Image.fromarray(probe, "L").filter(ImageFilter.MaxFilter(5)))
    return bool(out[2, 2] == 255 and out[0, 2] == 255)  # a corner pixel spreads to the window around it


def test_fast_closing_equals_pillow_on_structured_masks():
    from services.severity import _close

    mask = np.zeros((288, 384), dtype=bool)
    mask[40:200, 60:300] = True      # a leaf-like blob
    mask[90:110, 120:140] = False    # a hole
    mask[0:5, :] = True              # touching the border
    mask[:, 380:] = True
    for size in (7, 15):
        fast, reference = _close(mask, size), _pillow_close(mask, size)
        r = size // 2
        assert np.array_equal(fast[r:-r, r:-r], reference[r:-r, r:-r])
        if _pillow_clips_at_the_border():
            assert np.array_equal(fast, reference)


def test_estimate_is_unchanged_by_the_faster_implementation(monkeypatch):
    import services.severity as sv

    rng = np.random.default_rng(5)
    arr = np.zeros((420, 560, 3), dtype=np.float32)
    arr[:] = (70, 140, 80)
    arr[100:200, 150:300] = (200, 190, 60)  # yellow patch
    arr[250:300, 350:450] = (130, 90, 50)   # brown patch
    arr += rng.normal(0, 6, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    fast = sv.estimate(img)
    monkeypatch.setattr(sv, "_close", _pillow_close)
    slow = sv.estimate(img)
    assert fast.band == slow.band and fast.affected_percent > 5
    if _pillow_clips_at_the_border():
        assert fast.affected_percent == slow.affected_percent and fast.leaf_fraction == slow.leaf_fraction


def test_severity_png_is_lossless():
    import base64

    import services.severity as sv

    img = Image.fromarray(np.random.default_rng(2).integers(0, 255, (60, 80, 3), dtype=np.uint8))
    uri = sv._encode(img, "PNG")
    decoded = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert np.array_equal(np.asarray(decoded), np.asarray(img))
