"""Inference-service behaviour: request ids, timings, decisions, compact and batch modes,
caching, the ONNX Runtime fast path, model/stats endpoints and the API specification."""
import io
import json
import logging
import re

import numpy as np
import pytest
from PIL import Image

from config import BASE_DIR, TestConfig
from services.cache import ResultCache, fingerprint
from services.onnx_export import export
from services.predictor import Predictor
from services.similarity_client import UNAVAILABLE
from services.telemetry import Metrics, Stopwatch
from tests.conftest import FakeLeafCheck, image_bytes, leafy_bytes
from tests.test_app import FixedPredictor


def post(client, data, path="/api/predict", name="leaf.jpg", **kw):
    return client.post(path, data={"image": (io.BytesIO(data), name)}, content_type="multipart/form-data", **kw)


def post_batch(client, files, path="/api/predict/batch", **kw):
    data = {"images": [(io.BytesIO(b), n) for n, b in files]}
    return client.post(path, data=data, content_type="multipart/form-data", **kw)


class CachingConfig(TestConfig):
    PREDICTION_CACHE_SIZE = 8


class SearchOk(FakeLeafCheck):
    def search(self, data, filename="image", mimetype="application/octet-stream"):
        self.search_calls += 1
        return {"available": True, "results": [], "timing_ms": {}}


class SearchDown(FakeLeafCheck):
    def search(self, data, filename="image", mimetype="application/octet-stream"):
        self.search_calls += 1
        return {"available": False, "reason": UNAVAILABLE}


@pytest.fixture(scope="module")
def onnx_file(loaded_predictor, model_file, tmp_path_factory):
    return export(loaded_predictor.load(), tmp_path_factory.mktemp("onnx") / "model1.onnx", source=model_file)


@pytest.fixture
def onnx_predictor(model_file, onnx_file):
    p = Predictor(model_file, onnx_path=onnx_file)
    p.load()
    return p


# ------------------------------------------------------------ request ids & errors
def test_every_response_has_a_request_id_header(make_client):
    r = make_client().get("/api/health")
    assert re.fullmatch(r"[0-9a-f]{16}", r.headers["X-Request-ID"])


def test_a_valid_caller_request_id_is_echoed_and_a_bad_one_replaced(make_client):
    c = make_client()
    assert c.get("/api/health", headers={"X-Request-ID": "trace-1234.abcd"}).headers["X-Request-ID"] == "trace-1234.abcd"
    for bad_id in ("bad id with spaces", "short", "x" * 65, "<script>alert(1)</script>"):
        got = c.get("/api/health", headers={"X-Request-ID": bad_id}).headers["X-Request-ID"]
        assert re.fullmatch(r"[0-9a-f]{16}", got), bad_id


def test_input_errors_say_rejected_and_carry_the_request_id(make_client):
    r = post(make_client(), b"not an image", headers={"X-Request-ID": "trace-0001-abcd"})
    assert r.status_code == 422
    assert r.json["decision"] == "rejected" and r.json["request_id"] == "trace-0001-abcd" and r.json["success"] is False


def test_server_side_failures_say_error_not_rejected(make_client):
    r = post(make_client(), leafy_bytes())  # no model installed
    assert r.status_code == 503 and r.json["decision"] == "error"


def test_refusals_are_rejected_with_a_reason(make_client):
    r = post(make_client(leaf=False), leafy_bytes())
    assert r.status_code == 422 and r.json["decision"] == "rejected" and r.json["status"] == "not_cauliflower_leaf"


# ---------------------------------------------------------------- decisions
@pytest.mark.parametrize("probs,decision,reliable", [
    ([0.97, 0.01, 0.01, 0.01], "accepted", True),
    ([0.85, 0.10, 0.03, 0.02], "accepted", True),
    ([0.45, 0.30, 0.15, 0.10], "uncertain", False),
])
def test_decision_follows_the_confidence_threshold(make_client, probs, decision, reliable):
    r = post(make_client(predictor=FixedPredictor(probs)), leafy_bytes())
    assert r.status_code == 200
    assert r.json["decision"] == decision and r.json["result"]["reliable"] is reliable
    assert r.json["result"]["label"] == r.json["prediction"]["disease"]
    assert r.json["decision_reason"]


def test_top_k_controls_the_ranked_list(make_client):
    c = make_client(predictor=FixedPredictor([0.6, 0.25, 0.1, 0.05]))
    top = post(c, leafy_bytes(), query_string={"top_k": "2"}).json["result"]["top_k"]
    assert [t["label"] for t in top] == ["Bacterial spot rot", "Black Rot"]
    assert len(post(c, leafy_bytes()).json["result"]["top_k"]) == 3
    for bad in ("0", "9", "x", "-1"):
        r = post(c, leafy_bytes(), query_string={"top_k": bad})
        assert r.status_code == 400 and r.json["code"] == "invalid_parameter"


# ---------------------------------------------------------------- timings
def test_full_response_has_a_stage_timing_breakdown(loaded_predictor, make_client):
    r = post(make_client(predictor=loaded_predictor), leafy_bytes())
    t = r.json["timing_ms"]
    for stage in ("read_upload", "quality_gate", "leaf_check", "decode", "preprocess", "inference",
                  "explanation_render", "severity", "similarity_wait", "total"):
        assert stage in t and t[stage] >= 0, stage
    assert t["total"] >= t["inference"]
    assert r.json["inference"] == {"backend": "tensorflow", "model": "model1.h5"}
    assert "timing_ms" not in r.json["prediction"] and "backend" not in r.json["prediction"]
    assert r.json["cached"] is False


def test_stopwatch_accumulates_stages():
    sw = Stopwatch()
    with sw.stage("a"):
        pass
    sw.add("a", 2.0)
    sw.merge({"b": 1.5, "skip": "x"})
    d = sw.as_dict()
    assert d["a"] >= 2.0 and d["b"] == 1.5 and "skip" not in d and d["total"] >= 0


# ---------------------------------------------------------------- compact mode
def test_compact_mode_skips_the_extra_analyses(loaded_predictor, make_client):
    sim = FakeLeafCheck(True)
    c = make_client(predictor=loaded_predictor, similarity=sim)
    r = post(c, leafy_bytes(), query_string={"compact": "1"})
    assert r.status_code == 200
    body = r.json
    assert set(body["result"]) == {"label", "confidence", "confidence_percent", "confidence_level", "reliable", "top_k"}
    assert body["decision"] in {"accepted", "uncertain"}
    for heavy in ("prediction", "severity", "similar_images", "disease", "similar", "reasons"):
        assert heavy not in body
    assert sim.search_calls == 0 and sim.verify_calls == 1
    stages = body["timing_ms"]
    assert {"read_upload", "quality_gate", "leaf_check", "preprocess", "inference", "total"} <= set(stages)
    assert not {"decode", "severity", "similarity_wait", "explanation_render"} & set(stages)  # nothing decoded twice
    assert len(r.get_data()) < 3000


def test_compact_and_full_agree_on_the_prediction(loaded_predictor, make_client):
    c = make_client(predictor=loaded_predictor)
    full = post(c, leafy_bytes()).json
    compact = post(c, leafy_bytes(), query_string={"compact": "1"}).json
    assert compact["result"]["label"] == full["result"]["label"]
    assert compact["result"]["confidence"] == pytest.approx(full["result"]["confidence"], abs=1e-3)


def test_explain_off_returns_no_gradcam(loaded_predictor, make_client):
    r = post(make_client(predictor=loaded_predictor), leafy_bytes(), query_string={"explain": "0"})
    assert r.status_code == 200 and "explanation" not in r.json["prediction"]


# ---------------------------------------------------------------- ONNX fast path
def test_onnx_model_matches_keras(loaded_predictor, onnx_file):
    import onnxruntime as ort

    x = np.random.default_rng(3).uniform(0, 255, (5, 256, 256, 3)).astype(np.float32)
    expected = loaded_predictor.load()(x, training=False).numpy()
    got = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"]).run(None, {"image": x})[0]
    assert np.abs(expected - got).max() < 1e-5
    assert (expected.argmax(1) == got.argmax(1)).all()


def test_onnx_serves_plain_predictions_and_gradcam(onnx_predictor):
    assert onnx_predictor.onnx_status.startswith("active") and "Grad-CAM on ONNX" in onnx_predictor.onnx_status
    assert onnx_predictor.backend == "onnxruntime" and onnx_predictor.gradcam_backend == "onnxruntime"
    img = Image.open(io.BytesIO(leafy_bytes())).convert("RGB")
    plain = onnx_predictor.predict(img, explain=False)
    explained = onnx_predictor.predict(img, explain=True)
    assert plain["backend"] == explained["backend"] == "onnxruntime"
    assert "explanation" in explained and "explanation" not in plain
    assert explained["explanation"]["layer"].startswith("conv2d")
    assert plain["disease"] == explained["disease"]
    assert plain["confidence"] == pytest.approx(explained["confidence"], abs=1e-6)


def test_gradcam_from_onnx_activations_matches_keras(onnx_predictor):
    rng = np.random.default_rng(11)
    for _ in range(6):
        x = rng.uniform(0, 255, (1, 256, 256, 3)).astype(np.float32)
        probs_keras, cam_keras = onnx_predictor._forward(onnx_predictor._model, x, True)
        probs_onnx, cam_onnx = onnx_predictor._cam(onnx_predictor._onnx, onnx_predictor._gradcam, x)
        assert np.abs(probs_keras - probs_onnx).max() < 1e-5
        assert cam_keras.shape == cam_onnx.shape
        assert np.abs(cam_keras - cam_onnx).max() < 1e-4


def test_gradcam_math_on_a_hand_checked_case():
    from services.gradcam_onnx import cam_from_features

    conv = np.zeros((4, 4, 2), dtype=np.float32)
    conv[1, 1, 0], conv[2, 2, 1] = 2.0, 3.0
    hidden = np.array([1.0, 0.0], dtype=np.float32)            # second hidden unit is inactive (ReLU gradient 0)
    w_out = np.array([[0.0, 5.0], [9.0, 9.0]], dtype=np.float32)  # hidden -> 2 classes
    w_hidden = np.zeros((2 * 2 * 2, 2), dtype=np.float32)      # flat (2x2x2 pooled) -> hidden
    w_hidden[0, 0] = 1.0   # pooled cell (0,0), channel 0 feeds hidden unit 0
    w_hidden[3, 0] = -1.0  # pooled cell (0,1), channel 1 feeds hidden unit 0 negatively
    cam = cam_from_features(conv, hidden, top=1, w_hidden=w_hidden, w_out=w_out, pool=2)
    # grad wrt flat = w_hidden @ (w_out[:, 1] * [1, 0]) = w_hidden @ [5, 0]: channel 0 gets +5, channel 1 gets -5
    # weights = sum over pooled cells / (4 * 4) -> [5/16, -5/16]; cam = relu(2*5/16 at (1,1), -3*5/16 at (2,2))
    assert cam[1, 1] == pytest.approx(1.0) and cam[2, 2] == 0.0 and cam.sum() == pytest.approx(1.0)
    assert not cam_from_features(np.zeros((4, 4, 2), np.float32), hidden, 0, w_hidden, w_out, 2).any()


def test_a_missing_sidecar_leaves_gradcam_on_keras(model_file, onnx_file, tmp_path):
    import shutil

    shutil.copy(onnx_file, tmp_path / "model1.onnx")  # no .gradcam.npz next to it
    p = Predictor(model_file, onnx_path=tmp_path / "model1.onnx")
    p.load()
    assert p.backend == "onnxruntime" and p.gradcam_backend == "tensorflow" and "Grad-CAM on Keras" in p.onnx_status
    img = Image.open(io.BytesIO(leafy_bytes())).convert("RGB")
    assert p.predict(img, explain=False)["backend"] == "onnxruntime"
    assert p.predict(img, explain=True)["backend"] == "tensorflow"


def test_api_reports_the_onnx_backend(onnx_predictor, make_client):
    c = make_client(predictor=onnx_predictor)
    assert post(c, leafy_bytes(), query_string={"compact": "1"}).json["inference"]["backend"] == "onnxruntime"
    full = post(c, leafy_bytes()).json
    assert full["inference"]["backend"] == "onnxruntime" and full["prediction"]["explanation"]["method"] == "Grad-CAM"
    health = c.get("/api/health").json
    assert health["inference_backend"] == "onnxruntime" and health["gradcam_backend"] == "onnxruntime"


# ---------------------------------------------------------------- TensorFlow-free serving
def test_backend_names_are_validated(model_file):
    with pytest.raises(ValueError):
        Predictor(model_file, backend="bogus")


def test_onnx_only_mode_loads_without_keras(model_file, onnx_file):
    p = Predictor(model_file, onnx_path=onnx_file, backend="onnx")
    assert not p.is_loaded
    p.load()
    assert p.is_loaded and p._model is None and p.backend == "onnxruntime" and p.gradcam_backend == "onnxruntime"
    assert "TensorFlow not used" in p.onnx_status and "matches the model file" in p.onnx_status
    img = Image.open(io.BytesIO(leafy_bytes())).convert("RGB")
    r = p.predict(img, explain=True)
    assert r["backend"] == "onnxruntime" and r["explanation"]["method"] == "Grad-CAM"
    only = p.predict_array(p.preprocess(img))
    assert only["disease"] == r["disease"]
    batch, info = p.predict_batch([p.preprocess(img), p.preprocess(img)])
    assert len(batch) == 2 and info["backend"] == "onnxruntime"


def test_onnx_only_mode_never_imports_tensorflow(model_file, onnx_file):
    import subprocess
    import sys

    code = """
import io, json, sys
sys.modules["tensorflow"] = None  # any `import tensorflow` now fails
import numpy as np
from PIL import Image
from app import create_app
from config import TestConfig
from services.predictor import Predictor
from services.similarity_client import SimilarityClient


class Sim(SimilarityClient):
    def __init__(self):
        super().__init__("")

    def verify(self, data, filename="image", mimetype=""):
        return {"checked": True, "is_cauliflower_leaf": True, "score": 0.9, "threshold": 0.5}


rng = np.random.default_rng(1)
arr = np.stack([60 + rng.normal(0, 12, (300, 300)), 140 + rng.normal(0, 12, (300, 300)), 70 + rng.normal(0, 12, (300, 300))], -1)
buf = io.BytesIO()
Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(buf, format="JPEG")
app = create_app(TestConfig, predictor=Predictor(sys.argv[1], onnx_path=sys.argv[2], backend="onnx"), similarity=Sim())
c = app.test_client()
full = c.post("/api/predict", data={"image": (io.BytesIO(buf.getvalue()), "leaf.jpg")}, content_type="multipart/form-data")
compact = c.post("/api/predict?compact=1", data={"image": (io.BytesIO(buf.getvalue()), "leaf.jpg")}, content_type="multipart/form-data")
health = c.get("/api/health").json
print(json.dumps({"full": full.status_code, "compact": compact.status_code, "backend": full.json["inference"]["backend"],
                  "gradcam": "explanation" in full.json["prediction"], "model_loaded": health["model_loaded"],
                  "file_present": health["model_file_present"],
                  "tensorflow_loaded": any(m.startswith("tensorflow") and v is not None for m, v in sys.modules.items())}))
"""
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code, str(model_file), str(onnx_file)],
                         capture_output=True, text=True, cwd=BASE_DIR, timeout=300)
    assert out.returncode == 0, out.stderr[-1500:]
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result == {"full": 200, "compact": 200, "backend": "onnxruntime", "gradcam": True, "model_loaded": True,
                      "file_present": True, "tensorflow_loaded": False}


def test_onnx_only_mode_refuses_a_model_exported_from_a_different_file(tmp_path, onnx_file):
    from tests.conftest import build_notebook_architecture

    other = tmp_path / "other.h5"
    build_notebook_architecture().save(other)
    p = Predictor(other, onnx_path=onnx_file, backend="onnx")
    from services.predictor import ModelUnavailableError

    with pytest.raises(ModelUnavailableError, match="was not exported from"):
        p.load()
    assert not p.is_loaded


def test_onnx_only_mode_refuses_an_export_without_a_recorded_source(loaded_predictor, model_file, tmp_path):
    from services.predictor import ModelUnavailableError

    path = export(loaded_predictor.load(), tmp_path / "unsourced.onnx")  # no source recorded
    with pytest.raises(ModelUnavailableError):
        Predictor(model_file, onnx_path=path, backend="onnx").load()


def test_onnx_only_mode_without_the_onnx_file_is_a_clean_503(model_file, tmp_path, make_client):
    p = Predictor(model_file, onnx_path=tmp_path / "missing.onnx", backend="onnx")
    c = make_client(predictor=p)
    r = post(c, leafy_bytes())
    assert r.status_code == 503 and r.json["code"] == "model_unavailable"
    assert "missing.onnx" not in r.get_data(as_text=True)
    assert c.get("/api/health").json["model_file_present"] is False


def test_a_sidecar_from_another_export_is_ignored(model_file, onnx_file, tmp_path):
    import shutil

    shutil.copy(onnx_file, tmp_path / "model1.onnx")
    side = np.load(onnx_file.with_name("model1.gradcam.npz"))
    np.savez(tmp_path / "model1.gradcam.npz", **{**{k: side[k] for k in side.files}, "source_sha256": np.array("0" * 64)})
    p = Predictor(model_file, onnx_path=tmp_path / "model1.onnx", backend="onnx")
    p.load()
    assert p.gradcam_backend == "none" and "Grad-CAM unavailable" in p.onnx_status
    r = p.predict(Image.open(io.BytesIO(leafy_bytes())).convert("RGB"), explain=True)
    assert r["backend"] == "onnxruntime" and "explanation" not in r  # still classifies, without a heat-map


def test_a_stale_onnx_file_is_rejected(model_file, onnx_file, tmp_path):
    import onnx
    from onnx import numpy_helper

    proto = onnx.load(str(onnx_file))
    last_bias = [i for i in proto.graph.initializer if i.name.startswith("dense_b")][-1]
    arr = numpy_helper.to_array(last_bias).copy()
    arr[0] += 5.0
    last_bias.CopyFrom(numpy_helper.from_array(arr, last_bias.name))
    stale = tmp_path / "stale.onnx"
    onnx.save(proto, str(stale))

    p = Predictor(model_file, onnx_path=stale)
    p.load()
    assert p.onnx_status.startswith("rejected") and p.backend == "tensorflow"
    assert p.predict(Image.open(io.BytesIO(leafy_bytes())).convert("RGB"), explain=False)["backend"] == "tensorflow"


def test_missing_or_broken_onnx_falls_back_to_keras(model_file, tmp_path):
    missing = Predictor(model_file, onnx_path=tmp_path / "nope.onnx")
    missing.load()
    assert missing.backend == "tensorflow" and "not found" in missing.onnx_status
    broken = tmp_path / "broken.onnx"
    broken.write_bytes(b"not a model")
    p = Predictor(model_file, onnx_path=broken)
    p.load()
    assert p.backend == "tensorflow" and p.onnx_status.startswith("unavailable")


# ---------------------------------------------------------------- batch
def test_batch_mixes_accepted_and_rejected_images(onnx_predictor, make_client):
    c = make_client(predictor=onnx_predictor)
    r = post_batch(c, [("a.jpg", leafy_bytes()), ("broken.jpg", b"not an image"),
                       ("c.jpg", leafy_bytes(size=(320, 320))), ("d.gif", b"GIF89a....")])
    assert r.status_code == 200
    j = r.json
    assert j["count"] == 4 and j["accepted"] + j["uncertain"] == 2 and j["rejected"] == 2
    assert [x["index"] for x in j["results"]] == [0, 1, 2, 3]
    assert j["results"][1]["code"] == "corrupt_image" and j["results"][1]["decision"] == "rejected"
    assert j["results"][3]["code"] == "unsupported_extension"
    assert j["inference"]["backend"] == "onnxruntime" and j["inference"]["batched_forward_pass_ms"] > 0
    good = j["results"][0]
    assert good["success"] and set(good["result"]) >= {"label", "confidence", "top_k"}
    # Same answer as the single-image endpoint.
    single = post(c, leafy_bytes(), query_string={"compact": "1"}).json["result"]
    assert good["result"]["label"] == single["label"]
    assert good["result"]["confidence"] == pytest.approx(single["confidence"], abs=1e-4)


def test_batch_refuses_non_leaf_images_individually(loaded_predictor, make_client):
    r = post_batch(make_client(predictor=loaded_predictor, leaf=False), [("a.jpg", leafy_bytes())])
    assert r.status_code == 200 and r.json["rejected"] == 1
    assert r.json["results"][0]["code"] == "not_cauliflower_leaf" and "inference" in r.json


def test_batch_validation(loaded_predictor, make_client):
    class Small(TestConfig):
        BATCH_MAX_IMAGES = 2

    c = make_client(predictor=loaded_predictor, config=Small)
    empty = c.post("/api/predict/batch", data={}, content_type="multipart/form-data")
    assert empty.status_code == 400 and empty.json["code"] == "no_file"
    many = post_batch(c, [(f"{i}.jpg", leafy_bytes()) for i in range(3)])
    assert many.status_code == 400 and many.json["code"] == "too_many_images"


def test_batch_without_a_model_is_a_clean_503(make_client):
    r = post_batch(make_client(), [("a.jpg", leafy_bytes())])
    assert r.status_code == 503 and r.json["code"] == "model_unavailable"


def test_a_batch_costs_one_rate_limit_hit_per_image(loaded_predictor, make_client):
    class Tight(TestConfig):
        RATE_LIMIT_PREDICT = "3/minute"

    c = make_client(predictor=loaded_predictor, config=Tight)
    r = post_batch(c, [(f"{i}.jpg", leafy_bytes()) for i in range(4)])
    assert r.status_code == 429 and r.json["code"] == "rate_limited"
    ok = post_batch(make_client(predictor=loaded_predictor, config=Tight), [(f"{i}.jpg", leafy_bytes()) for i in range(2)])
    assert ok.status_code == 200


def test_single_uploads_stay_limited_to_one_image(loaded_predictor, make_client):
    class OneImage(TestConfig):
        MAX_UPLOAD_BYTES = 100_000
        MAX_CONTENT_LENGTH = 5_000_000

    c = make_client(predictor=loaded_predictor, config=OneImage)
    big = image_bytes(size=(400, 400))  # about 480 KB
    assert len(big) > 100_000
    r = post(c, big, name="leaf.png")
    assert r.status_code == 413 and r.json["code"] == "file_too_large"
    assert post_batch(c, [("big.png", big)]).json["results"][0]["code"] == "file_too_large"


# ---------------------------------------------------------------- caching
def test_a_repeated_image_is_served_from_the_cache(loaded_predictor, make_client):
    sim = SearchOk(True)
    c = make_client(predictor=loaded_predictor, config=CachingConfig, similarity=sim)
    first = post(c, leafy_bytes())
    second = post(c, leafy_bytes())
    assert first.json["cached"] is False and second.json["cached"] is True
    assert sim.verify_calls == 1 and sim.search_calls == 1
    assert second.json["result"] == first.json["result"]
    assert second.json["request_id"] != first.json["request_id"]
    assert second.json["timing_ms"]["total"] < first.json["timing_ms"]["total"]
    stats = c.get("/api/stats").json["cache"]
    assert stats["hits"] == 1 and stats["misses"] == 1 and stats["hit_rate"] == 0.5


def test_modes_and_images_are_cached_separately(loaded_predictor, make_client):
    sim = SearchOk(True)
    c = make_client(predictor=loaded_predictor, config=CachingConfig, similarity=sim)
    post(c, leafy_bytes())
    assert post(c, leafy_bytes(), query_string={"compact": "1"}).json["cached"] is False
    assert post(c, leafy_bytes(size=(310, 310))).json["cached"] is False


def test_a_temporary_similarity_failure_is_not_remembered(loaded_predictor, make_client):
    sim = SearchDown(True)
    c = make_client(predictor=loaded_predictor, config=CachingConfig, similarity=sim)
    assert post(c, leafy_bytes()).json["cached"] is False
    assert post(c, leafy_bytes()).json["cached"] is False
    assert sim.verify_calls == 2


def test_refusals_are_never_cached(loaded_predictor, make_client):
    sim = FakeLeafCheck(False)
    c = make_client(predictor=loaded_predictor, config=CachingConfig, similarity=sim)
    post(c, leafy_bytes())
    post(c, leafy_bytes())
    assert sim.verify_calls == 2


def test_result_cache_is_a_bounded_lru():
    cache = ResultCache(2)
    cache.put("a", {"v": 1})
    cache.put("b", {"v": 2})
    assert cache.get("a") == {"v": 1}  # refreshes "a"
    cache.put("c", {"v": 3})
    assert cache.get("b") is None and cache.get("a") and cache.get("c")
    cache.get("a")["v"] = 99  # callers get copies
    assert cache.get("a") == {"v": 1}
    assert cache.stats()["items"] == 2
    off = ResultCache(0)
    off.put("a", {"v": 1})
    assert off.get("a") is None and not off.enabled


def test_fingerprint_depends_on_bytes_and_mode():
    assert fingerprint(b"x", "full") != fingerprint(b"y", "full") != fingerprint(b"y", "compact")
    assert fingerprint(b"x", "full") == fingerprint(b"x", "full")


# ---------------------------------------------------------------- model info & stats
def test_model_endpoint_exposes_the_evaluation(make_client):
    j = make_client().get("/api/model").json
    assert j["success"] and j["model"]["classes"][1] == "Black Rot" and j["model"]["parameters"] == 183812
    ev = j["evaluation"]
    assert ev["images"] == 99 and ev["accuracy"] == pytest.approx(0.8788, abs=1e-4)
    assert len(ev["confusion_matrix"]["rows_true_cols_pred"]) == 4
    assert sum(map(sum, ev["confusion_matrix"]["rows_true_cols_pred"])) == 99
    assert j["reliability"]["uncertain_below"] == 0.8
    assert j["serving"]["mode"] == "auto" and j["serving"]["gradcam_backend"] == "none"
    assert j["serving"]["model_loaded"] is False
    assert j["evaluation"]["set"].startswith("held-out")


def test_stats_endpoint_counts_requests_decisions_and_latency(make_client):
    c = make_client(predictor=FixedPredictor([0.45, 0.30, 0.15, 0.10]))
    post(c, leafy_bytes())
    post(c, b"not an image")
    c.get("/api/health")  # not counted
    j = c.get("/api/stats").json
    assert j["requests_total"] == 2
    assert j["decisions"] == {"uncertain": 1, "rejected": 1}
    assert j["by_status"] == {"200": 1, "422": 1}
    lat = j["endpoints"]["POST /api/predict"]["latency_ms"]
    assert lat["p50"] > 0 and lat["p95"] >= lat["p50"]
    assert j["confidence_levels"] == {"uncertain": 1}


def test_metrics_percentiles():
    m = Metrics()
    for ms in range(1, 101):
        m.record("GET /x", 200, float(ms))
    lat = m.snapshot()["endpoints"]["GET /x"]["latency_ms"]
    assert lat["p50"] in (50.0, 51.0) and lat["p95"] in (95.0, 96.0) and lat["max"] == 100.0


# ---------------------------------------------------------------- logging
def test_request_log_is_structured_and_free_of_user_data(loaded_predictor, make_client, caplog):
    c = make_client(predictor=loaded_predictor)
    with caplog.at_level(logging.INFO, logger="caulicare.request"):
        post(c, leafy_bytes(), name="my-secret-field-photo.jpg", headers={"X-Request-ID": "trace-log-0001"})
    line = json.loads([r.message for r in caplog.records if r.name == "caulicare.request"][-1])
    assert line["request_id"] == "trace-log-0001" and line["status"] == 200 and line["path"] == "/api/predict"
    assert line["decision"] in {"accepted", "uncertain"} and line["latency_ms"] > 0 and "disease" in line
    assert "my-secret-field-photo" not in json.dumps(line)


# ---------------------------------------------------------------- specification
def _spec_text():
    return (BASE_DIR / "docs" / "openapi.yaml").read_text(encoding="utf-8")


def test_openapi_is_served(make_client):
    r = make_client().get("/api/openapi.yaml")
    assert r.status_code == 200 and r.data.decode().startswith("openapi: 3.0")


def test_every_api_route_is_documented(make_client):
    app = make_client().application
    documented = set(re.findall(r"^  (/api/[^\s:]*):", _spec_text(), flags=re.M))
    routes = {re.sub(r"<[^>]+>", "{name}", rule.rule) for rule in app.url_map.iter_rules()
              if rule.rule.startswith("/api/")}
    assert routes <= documented, f"undocumented: {sorted(routes - documented)}"


def test_every_documented_stage_is_a_real_timing_key():
    spec = _spec_text()
    for stage in ("read_upload", "quality_gate", "leaf_check", "decode", "preprocess", "inference",
                  "explanation_render", "severity", "similarity_wait", "total"):
        assert stage in spec


# ---------------------------------------------------------------- developer page & exporter
def test_developer_page_documents_the_decision_values(make_client):
    r = make_client().get("/developers")
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    for word in ("accepted", "uncertain", "rejected", "compact=1", "/api/predict/batch", "X-Request-ID"):
        assert word in html
    assert 'href="/developers"' in make_client().get("/").get_data(as_text=True)


def test_exporter_refuses_layers_it_cannot_translate_exactly(tmp_path):
    import tensorflow as tf

    from services.onnx_export import UnsupportedModelError, build_onnx

    model = tf.keras.Sequential([tf.keras.Input((256, 256, 3)), tf.keras.layers.Conv2D(4, 3, padding="same"),
                                 tf.keras.layers.Flatten(), tf.keras.layers.Dense(4, activation="softmax")])
    with pytest.raises(UnsupportedModelError):
        build_onnx(model)


# ---------------------------------------------------------------- percentages
def test_scores_come_as_fractions_and_percentages(make_client):
    c = make_client(predictor=FixedPredictor([0.9871, 0.0102, 0.0021, 0.0006]))
    j = post(c, leafy_bytes(), query_string={"top_k": "4"}).json
    assert j["result"]["confidence"] == 0.9871 and j["result"]["confidence_percent"] == 98.7
    assert [t["percent"] for t in j["result"]["top_k"]] == [98.7, 1.0, 0.2, 0.1]
    assert j["prediction"]["all_predictions_percent"][j["result"]["label"]] == 98.7
    assert sum(j["prediction"]["all_predictions_percent"].values()) == pytest.approx(100, abs=0.3)
    compact = post(c, leafy_bytes(), query_string={"compact": "1"}).json["result"]
    assert compact["confidence_percent"] == 98.7 and compact["top_k"][0]["percent"] == 98.7


# ---------------------------------------------------------------- second opinion
def _neighbours(*labels):
    return {"available": True, "results": [{"id": f"r{i}", "label": lab, "similarity": 0.8, "thumbnail": ""}
                                           for i, lab in enumerate(labels)], "timing_ms": {}}


class Neighbours(FakeLeafCheck):
    def __init__(self, *labels):
        super().__init__(True)
        self.labels = labels

    def search(self, data, filename="image", mimetype="application/octet-stream"):
        self.search_calls += 1
        return _neighbours(*self.labels)


DOWNY = [0.0005, 0.0005, 0.999, 0.0]  # a confident Downy Mildew prediction


def test_a_confident_prediction_contradicted_by_the_nearest_photos_is_flagged_unreliable(make_client):
    c = make_client(predictor=FixedPredictor(DOWNY), similarity=Neighbours(*["No disease"] * 5))
    j = post(c, leafy_bytes()).json
    assert j["prediction"]["disease"] == "Downy Mildew"          # the label itself is never changed
    assert j["decision"] == "uncertain" and j["result"]["reliable"] is False
    assert j["second_opinion"] == {"available": True, "neighbours": 5, "votes": 5, "consensus_label": "No disease",
                                   "agrees": False}
    assert "5 of the 5 most similar" in j["decision_reason"] and "No disease" in j["decision_reason"]


def test_agreement_or_a_split_vote_leaves_the_decision_alone(make_client):
    agree = post(make_client(predictor=FixedPredictor(DOWNY), similarity=Neighbours(*["Downy Mildew"] * 5)), leafy_bytes()).json
    assert agree["decision"] == "accepted" and agree["second_opinion"]["agrees"] is True
    split = post(make_client(predictor=FixedPredictor(DOWNY),
                             similarity=Neighbours("No disease", "No disease", "Black Rot", "Downy Mildew", "Black Rot")),
                 leafy_bytes()).json
    assert split["decision"] == "accepted" and split["second_opinion"]["agrees"] is None


def test_no_second_opinion_without_the_similarity_service(make_client):
    j = post(make_client(predictor=FixedPredictor(DOWNY)), leafy_bytes()).json
    assert j["second_opinion"] == {"available": False} and j["decision"] == "accepted"


def test_an_already_uncertain_result_keeps_its_own_reason():
    from services.analysis import apply_second_opinion, second_opinion
    from services.predictor import summarize

    pred = summarize([0.5, 0.2, 0.2, 0.1])
    result = {"decision": "uncertain", "decision_reason": "own reason", "result": {"reliable": False}}
    out = apply_second_opinion(result, pred, second_opinion(pred, _neighbours(*["No disease"] * 4)))
    assert out["decision_reason"] == "own reason" and out["second_opinion"]["agrees"] is False


def test_result_page_script_flags_a_disagreement():
    js = (BASE_DIR / "static" / "js" / "result-view.js").read_text(encoding="utf-8")
    assert "second_opinion" in js and "Second opinion disagrees" in js
