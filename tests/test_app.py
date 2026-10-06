import io
import threading
import time

import numpy as np
import pytest
from PIL import Image, ImageFilter

from app import create_app
from config import BASE_DIR, TestConfig
from services.analysis import build_result, not_cauliflower_outcome, suitability_outcome
from services.image_validation import assess_quality, load_image
from services.knowledge_base import KnowledgeBase
from services.predictor import CLASS_NAMES, Predictor, confidence_tier, summarize
from tests.conftest import FakeLeafCheck, flat_bytes, image_bytes, leafy_bytes


def upload(client, data, name="leaf.png", mimetype=None, **kw):
    file = (io.BytesIO(data), name, mimetype) if mimetype else (io.BytesIO(data), name)
    return client.post("/api/predict", data={"image": file}, content_type="multipart/form-data", **kw)


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase(BASE_DIR / "data" / "diseases.json")


class FixedPredictor(Predictor):
    """Returns a chosen probability vector, to test result composition without TensorFlow."""

    def __init__(self, probs):
        super().__init__("unused.h5")
        self.probs = np.array(probs, dtype=np.float32)
        self._model = object()

    def _forward(self, model, batch, explain):
        return self.probs, None


# ------------------------------------------------------------------- pages
@pytest.mark.parametrize("path", ["/", "/analyze", "/diseases", "/diseases/black-rot", "/photo-guide",
                                  "/history", "/report", "/about"])
def test_pages_render(make_client, path):
    r = make_client().get(path)
    assert r.status_code == 200
    assert b"CauliCare" in r.data
    assert b"not a definitive agricultural diagnosis" in r.data


def test_unknown_page_404(make_client):
    r = make_client().get("/nope")
    assert r.status_code == 404 and b"Page not found" in r.data


def test_home_shows_honest_test_metrics(make_client):
    html = make_client().get("/").data.decode()
    assert "87.9%" in html and "0.891" in html and "96.9" not in html
    for name in ["Black Rot", "Downy Mildew", "Bacterial Spot Rot", "No Disease"]:
        assert name in html


def test_about_renders_confusion_matrix_from_metadata(make_client):
    html = make_client().get("/about").data.decode()
    assert "Confusion matrix" in html and "183,812" in html and "0.891" in html


# ------------------------------------------------------------- upload validation
def test_missing_upload(make_client):
    r = make_client().post("/api/predict", data={}, content_type="multipart/form-data")
    assert r.status_code == 400 and r.json["code"] == "no_file"


def test_invalid_extension(make_client):
    r = upload(make_client(), b"GIF89a....", name="leaf.gif")
    assert r.status_code == 415 and r.json["error"] == "Please upload a JPG, PNG, or WEBP image."


def test_invalid_mimetype(make_client):
    r = upload(make_client(), image_bytes(), name="leaf.png", mimetype="application/pdf")
    assert r.status_code == 415 and r.json["code"] == "unsupported_mimetype"


def test_generic_mimetype_is_checked_by_content(loaded_predictor, make_client):
    r = upload(make_client(predictor=loaded_predictor), leafy_bytes(), name="leaf.jpg", mimetype="application/octet-stream")
    assert r.status_code == 200 and r.json["status"] == "ok"


def test_corrupted_image(make_client):
    r = upload(make_client(), b"this is not an image", name="leaf.jpg")
    assert r.status_code == 422
    assert r.json["error"] == "We couldn't process this image. Please try another photo."


def test_truncated_image(make_client):
    r = upload(make_client(), leafy_bytes()[:600], name="leaf.jpg")
    assert r.status_code == 422 and r.json["code"] == "corrupt_image"


def test_format_mismatch_rejected(make_client):
    buf = io.BytesIO()
    Image.new("RGB", (200, 200)).save(buf, format="GIF")  # real GIF renamed to .png
    r = upload(make_client(), buf.getvalue(), name="leaf.png")
    assert r.status_code == 415 and r.json["code"] == "unsupported_format"


def test_empty_file(make_client):
    r = upload(make_client(), b"")
    assert r.status_code == 400 and r.json["code"] == "empty_file"


def test_oversized_image(make_client):
    client = make_client()
    client.application.config["MAX_CONTENT_LENGTH"] = 1024
    r = upload(client, image_bytes(size=(400, 400)))
    assert r.status_code == 413 and "too large" in r.json["error"] and r.json["code"] == "file_too_large"


def test_model_missing_returns_503_without_internals(make_client):
    r = upload(make_client(), leafy_bytes(), name="leaf.jpg")
    assert r.status_code == 503 and r.json["code"] == "model_unavailable"
    assert "Traceback" not in r.get_data(as_text=True) and "does/not/exist" not in r.get_data(as_text=True)


def test_unexpected_failure_is_a_clean_message(make_client):
    class Broken(Predictor):
        def predict(self, img, explain=True):
            raise RuntimeError("secret internal detail")

    r = upload(make_client(predictor=Broken("x.h5")), leafy_bytes(), name="leaf.jpg")
    assert r.status_code == 500 and "secret internal detail" not in r.get_data(as_text=True)
    assert r.json["error"] == "We couldn't process this image. Please try another photo."


# ------------------------------------------------------------ quality gate
def test_blank_image_refused(make_client):
    r = upload(make_client(), flat_bytes(128))
    assert r.status_code == 422 and r.json["status"] == "low_quality" and r.json["success"] is False
    assert "blank" in " ".join(r.json["reasons"]).lower()


def test_dark_image_refused(make_client):
    dark = np.clip(np.random.default_rng(0).normal(12, 6, (300, 300, 3)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(dark).save(buf, format="PNG")
    r = upload(make_client(), buf.getvalue())
    assert r.status_code == 422 and r.json["status"] == "low_quality"
    assert any("too dark" in m for m in r.json["reasons"])
    assert "prediction" not in r.json  # the classifier never ran


def test_overexposed_image_refused():
    arr = np.full((300, 300, 3), 255, dtype=np.uint8)
    arr[::4, ::4] = (90, 160, 80)  # a little detail so it is not "blank"
    q = assess_quality(Image.fromarray(arr))
    assert "overexposed" in [p.code for p in q.problems]


def test_blurry_image_refused():
    # Leaf-coloured image with large-scale structure (veins) plus fine detail.
    y, x = np.mgrid[0:600, 0:600]
    veins = (np.sin(x / 9.0) + np.sin((x + y) / 23.0)) * 40
    fine = np.random.default_rng(4).normal(0, 10, (600, 600))
    g = 140 + veins + fine
    arr = np.clip(np.stack([g * 0.45, g, g * 0.5], -1), 0, 255).astype(np.uint8)
    leaf = Image.fromarray(arr)
    assert assess_quality(leaf).acceptable
    q = assess_quality(leaf.filter(ImageFilter.GaussianBlur(14)))
    assert [p.code for p in q.problems] == ["too_blurry"]


def test_low_resolution_refused(make_client):
    r = upload(make_client(), leafy_bytes(size=(100, 90)), name="leaf.jpg")
    assert r.json["status"] == "low_quality"
    assert any("resolution is too low" in m for m in r.json["reasons"])


# ------------------------------------------------------ cauliflower-leaf gate
class ExplodingPredictor(Predictor):
    """Fails the test if the disease model is ever run."""

    def __init__(self):
        super().__init__("unused.h5")
        self._model = object()

    def _forward(self, model, batch, explain):
        raise AssertionError("the disease model must not run")


def test_rejected_image_never_reaches_the_disease_model(make_client):
    fake = FakeLeafCheck(False)
    r = upload(make_client(predictor=ExplodingPredictor(), similarity=fake), image_bytes(), name="cat.png")
    assert r.status_code == 422 and r.json["success"] is False
    assert r.json["status"] == r.json["code"] == "not_cauliflower_leaf"
    assert r.json["title"] == "Please upload a cauliflower leaf image"
    assert "does not appear to contain a cauliflower leaf" in r.json["message"]
    for key in ("prediction", "disease", "severity", "similar_images"):
        assert key not in r.json  # no disease, confidence, Grad-CAM, affected region or similar images
    assert r.json["leaf_check"]["is_cauliflower_leaf"] is False
    assert fake.verify_calls == 1 and fake.search_calls == 0  # stopped before similarity search


def test_unverifiable_image_is_refused_not_analysed(make_client):
    """Fail closed: when the leaf check cannot run, nothing is analysed."""
    fake = FakeLeafCheck(None)
    r = upload(make_client(predictor=ExplodingPredictor(), similarity=fake), leafy_bytes(), name="leaf.jpg")
    assert r.status_code == 503 and r.json["code"] == "leaf_check_unavailable"
    assert "prediction" not in r.json and fake.search_calls == 0


def test_unconfigured_similarity_service_refuses_analysis(loaded_predictor):
    from services.similarity_client import SimilarityClient

    client = create_app(TestConfig, predictor=loaded_predictor, similarity=SimilarityClient("")).test_client()
    r = upload(client, leafy_bytes(), name="leaf.jpg")
    assert r.status_code == 503 and r.json["code"] == "leaf_check_unavailable"


def test_quality_gate_runs_before_the_leaf_check(make_client):
    fake = FakeLeafCheck(True)
    r = upload(make_client(similarity=fake), flat_bytes(128))
    assert r.json["status"] == "low_quality" and fake.verify_calls == 0


def overexposed_bytes():
    arr = np.full((400, 400, 3), 255, dtype=np.uint8)
    arr[::5, :] = (20, 20, 20)  # lines of "text" on a white page
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_overexposed_non_leaf_gets_the_cauliflower_message(make_client):
    """A white document is "overexposed" to the quality gate, but the useful answer is
    that it is not a cauliflower leaf; an overexposed leaf still gets photo advice."""
    assert assess_quality(Image.open(io.BytesIO(overexposed_bytes()))).problems
    no = upload(make_client(predictor=ExplodingPredictor(), leaf=False), overexposed_bytes(), name="doc.png")
    assert no.status_code == 422 and no.json["status"] == "not_cauliflower_leaf"
    yes = upload(make_client(predictor=ExplodingPredictor(), leaf=True), overexposed_bytes(), name="leaf.png")
    assert yes.status_code == 422 and yes.json["status"] == "low_quality"
    down = upload(make_client(predictor=ExplodingPredictor(), leaf=None), overexposed_bytes(), name="leaf.png")
    assert down.json["status"] == "low_quality"  # still refused when the check is unavailable


def test_accepted_leaf_runs_the_full_pipeline(loaded_predictor, make_client):
    fake = FakeLeafCheck(True)
    r = upload(make_client(predictor=loaded_predictor, similarity=fake), leafy_bytes(), name="leaf.jpg")
    assert r.status_code == 200 and r.json["status"] == "ok"
    assert r.json["leaf_check"] == {"checked": True, "is_cauliflower_leaf": True, "score": 0.9, "threshold": 0.5}
    assert fake.verify_calls == 1 and fake.search_calls == 1


def verify_upload(client, data, name="leaf.jpg"):
    return client.post("/api/verify", data={"image": (io.BytesIO(data), name)}, content_type="multipart/form-data")


def test_verify_endpoint_accepts_and_rejects_without_running_the_model(make_client):
    ok = verify_upload(make_client(predictor=ExplodingPredictor(), leaf=True), leafy_bytes())
    assert ok.status_code == 200 and ok.json["status"] == "cauliflower_leaf" and "prediction" not in ok.json
    no = verify_upload(make_client(predictor=ExplodingPredictor(), leaf=False), image_bytes(), "x.png")
    assert no.status_code == 422 and no.json["status"] == "not_cauliflower_leaf"
    bad = verify_upload(make_client(), b"not an image", "x.png")
    assert bad.status_code == 422 and bad.json["code"] == "corrupt_image"


def test_verify_endpoint_is_rate_limited_with_predict():
    class Limited(TestConfig):
        RATE_LIMIT_PREDICT = "2/minute"

    c = create_app(Limited, predictor=Predictor("does/not/exist.h5"), similarity=FakeLeafCheck(False)).test_client()
    codes = [verify_upload(c, leafy_bytes()).status_code for _ in range(3)]
    assert codes == [422, 422, 429]


def test_leaf_gate_rejects_scenes_keeps_plants():
    # Warm-toned scene (skin, wood, brick tones) with texture: very few plant-coloured pixels.
    scene = np.zeros((480, 640, 3), dtype=np.uint8)
    scene[...] = (205, 150, 120)
    scene[240:, :] = (150, 70, 50)
    scene += np.random.default_rng(2).integers(0, 25, scene.shape, dtype=np.uint8)
    assert assess_quality(Image.fromarray(scene)).looks_like_leaf is False
    assert assess_quality(Image.open(io.BytesIO(leafy_bytes()))).looks_like_leaf is True


def test_refusal_outcomes():
    assert suitability_outcome(assess_quality(Image.open(io.BytesIO(leafy_bytes())))) is None
    assert suitability_outcome(assess_quality(Image.open(io.BytesIO(flat_bytes(128)))))["status"] == "low_quality"
    out = not_cauliflower_outcome({"checked": True, "is_cauliflower_leaf": False, "score": 0.02, "threshold": 0.5})
    assert out["status"] == "not_cauliflower_leaf" and len(out["reasons"]) == 2


# ---------------------------------------------------------------- prediction
@pytest.mark.parametrize("score,tier", [(0.5, "uncertain"), (0.7999, "uncertain"), (0.8, "moderate"),
                                        (0.9499, "moderate"), (0.95, "high"), (1.0, "high")])
def test_confidence_tiers(score, tier):
    assert confidence_tier(score) == tier


def test_summary_contains_probabilities_and_runner_up():
    s = summarize([0.6, 0.05, 0.3, 0.05])
    assert s["disease"] == "Bacterial spot rot" and s["confidence_level"] == "uncertain"
    assert s["confidence_message"].startswith("Low confidence")
    assert s["runner_up"] == {"disease": "Downy Mildew", "probability": 0.3}
    assert s["confusable_close_call"] is True
    assert list(s["all_predictions"])[:2] == ["Bacterial spot rot", "Downy Mildew"]
    assert set(s["all_predictions"]) == set(CLASS_NAMES)


def test_prediction_pipeline_end_to_end(loaded_predictor, make_client):
    r = upload(make_client(predictor=loaded_predictor), leafy_bytes(), name="leaf.jpg")
    assert r.status_code == 200 and r.json["status"] == "ok", r.json
    p = r.json["prediction"]
    assert p["disease"] in CLASS_NAMES
    assert set(p["all_predictions"]) == set(CLASS_NAMES)
    assert abs(sum(p["all_predictions"].values()) - 1) < 1e-2
    assert p["confidence"] == max(p["all_predictions"].values())
    assert p["confidence_level"] in {"uncertain", "moderate", "high"}
    assert r.json["disease"]["label"] == p["disease"]
    assert r.json["reasons"] and "definitive" in r.json["disclaimer"]


def test_gradcam_images_returned(loaded_predictor, make_client):
    p = upload(make_client(predictor=loaded_predictor), leafy_bytes(), name="leaf.jpg").json["prediction"]
    assert p["explanation"]["method"] == "Grad-CAM"
    assert p["explanation"]["heatmap"].startswith("data:image/jpeg;base64,")
    assert p["explanation"]["overlay"].startswith("data:image/jpeg;base64,")


def test_gradcam_failure_keeps_prediction(model_file, make_client):
    p = Predictor(model_file)
    p.load()
    p._last_conv_index = 10_000  # force the Grad-CAM path to fail
    r = upload(make_client(predictor=p), leafy_bytes(), name="leaf.jpg")
    assert r.status_code == 200 and r.json["status"] == "ok"
    assert "explanation" not in r.json["prediction"]


def test_prediction_is_deterministic(loaded_predictor):
    img = Image.open(io.BytesIO(leafy_bytes())).convert("RGB")
    a = loaded_predictor.predict(img, explain=False)
    b = loaded_predictor.predict(img, explain=True)
    assert a["all_predictions"] == b["all_predictions"]


def test_webp_upload(loaded_predictor, make_client):
    r = upload(make_client(predictor=loaded_predictor), leafy_bytes(fmt="WEBP"), name="leaf.webp")
    assert r.status_code == 200 and r.json["status"] == "ok"


def test_close_call_shows_comparison(make_client):
    r = upload(make_client(predictor=FixedPredictor([0.78, 0.01, 0.18, 0.03])), leafy_bytes(), name="leaf.jpg")
    body = r.json
    assert body["prediction"]["confusable_close_call"] is True
    names = [c["name"] for c in body["similar"]["conditions"]]
    assert names == ["Bacterial Spot Rot", "Downy Mildew"]
    assert "look similar" in body["similar"]["note"]
    assert any("strongest competing class was Downy Mildew" in r for r in body["reasons"])


def test_confident_black_rot_has_no_comparison(make_client):
    body = upload(make_client(predictor=FixedPredictor([0.001, 0.997, 0.001, 0.001])), leafy_bytes(), name="leaf.jpg").json
    assert body["prediction"]["confidence_level"] == "high" and body["similar"] is None


def test_confident_downy_mildew_still_compared_with_bacterial_spot(kb):
    q = assess_quality(Image.open(io.BytesIO(leafy_bytes())))
    result = build_result(summarize([0.002, 0.001, 0.996, 0.001]), q, kb)
    assert "Commonly confused" in result["similar"]["note"]
    assert result["similar"]["conditions"][1]["name"] == "Bacterial Spot Rot"


def test_preprocess_matches_training_contract():
    img = Image.fromarray(np.full((480, 640, 4), 200, dtype=np.uint8), "RGBA")
    arr = Predictor.preprocess(img)
    assert arr.shape == (1, 256, 256, 3) and arr.dtype == np.float32
    assert arr.max() > 1.0  # raw 0-255: the model's own Rescaling layer normalises


def test_numpy_resize_is_identical_to_tensorflow():
    import tensorflow as tf

    from services.predictor import resize_bilinear_like_tf
    rng = np.random.default_rng(3)
    for hw in [(64, 64), (100, 300), (255, 257), (256, 256), (513, 129), (1200, 900)]:
        rgb = rng.integers(0, 256, (*hw, 3), dtype=np.uint8)
        ref = tf.image.resize(rgb, (256, 256), method="bilinear").numpy()
        assert np.array_equal(resize_bilinear_like_tf(rgb), ref), hw


def test_exif_rotation_not_applied_to_model_input():
    img = Image.new("RGB", (300, 200), (10, 120, 30))
    exif = img.getexif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    out = load_image(buf.getvalue(), {"JPEG"}, 10**8)
    assert out.size == (300, 200) and out.info["exif_orientation"] == 6


def test_model_loads_once(model_file):
    p = Predictor(model_file)
    assert p.load() is p.load()


def test_relative_model_path_resolves_from_project_root():
    from config import Config
    assert Config.MODEL_PATH.is_absolute() and str(Config.MODEL_PATH).startswith(str(BASE_DIR))


def test_image_pipeline_is_serialised(make_client):
    """Concurrent uploads are processed one at a time (memory safety on small instances)."""
    p = FixedPredictor([0.7, 0.1, 0.1, 0.1])
    active, peak, lock = [0], [0], threading.Lock()
    original = p._forward

    def slow_forward(model, batch, explain):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return original(model, batch, explain)

    p._forward = slow_forward
    client = make_client(predictor=p)
    codes = []

    def post():
        codes.append(upload(client, leafy_bytes(), name="leaf.jpg").status_code)

    threads = [threading.Thread(target=post) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert codes == [200] * 4 and peak[0] == 1


# --------------------------------------------------------- knowledge / library
def test_diseases_api(make_client):
    r = make_client().get("/api/diseases")
    assert r.status_code == 200 and sorted(d["label"] for d in r.json["diseases"]) == sorted(CLASS_NAMES)


@pytest.mark.parametrize("key", ["black-rot", "Black Rot", "downy mildew"])
def test_disease_lookup(make_client, key):
    assert make_client().get(f"/api/diseases/{key}").status_code == 200


def test_disease_not_found(make_client):
    r = make_client().get("/api/diseases/unknown")
    assert r.status_code == 404 and r.json["success"] is False
    assert make_client().get("/diseases/unknown").status_code == 404


def test_disease_detail_page_shows_test_metrics(make_client):
    html = make_client().get("/diseases/downy-mildew").data.decode()
    assert "Model on this class" in html and "confused with Bacterial spot rot" in html


def test_library_search_ranks_knowledge(make_client):
    r = make_client().get("/api/search?q=yellow V-shaped lesions at the leaf margin")
    assert r.status_code == 200 and r.json["results"][0]["disease_id"] == "black-rot"
    assert make_client().get("/api/search?q=").status_code == 400


def test_compare_uses_knowledge_records_verbatim(kb):
    c = kb.compare("Bacterial spot rot", "Downy Mildew")
    dm = kb.get("downy-mildew")
    assert c["conditions"][1]["visual_signs"] == dm["visual_signs"]


# ------------------------------------------------------------ health / security
def test_health(make_client):
    r = make_client().get("/api/health")
    assert r.json["status"] == "ok" and r.json["model_loaded"] is False and r.json["classes"] == CLASS_NAMES
    assert set(r.json["confidence_tiers"]) == {"uncertain", "moderate", "high"}


def test_security_headers(make_client):
    r = make_client().get("/")
    csp = r.headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp and "frame-ancestors" in csp
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert "Referrer-Policy" in r.headers


def test_static_js_served_with_js_mimetype(make_client):
    for name in ["common.js", "analyze.js", "result-view.js", "history.js", "report.js", "diseases.js"]:
        r = make_client().get(f"/static/js/{name}")
        assert r.status_code == 200 and "javascript" in r.mimetype, name


def test_path_traversal_is_not_served(make_client):
    for path in ["/static/../config.py", "/static/..%2fconfig.py", "/diseases/..%2f..%2fapp.py"]:
        r = make_client().get(path)
        assert r.status_code == 404 and b"SECRET_KEY" not in r.data


def _client(**overrides):
    cfg = type("Cfg", (TestConfig,), overrides)
    return create_app(cfg, predictor=Predictor("does/not/exist.h5")).test_client()


def test_rate_limit():
    client = _client(RATE_LIMIT_PREDICT="2/minute")
    codes = [upload(client, b"x", name="a.gif").status_code for _ in range(3)]
    assert codes == [415, 415, 429]


def test_spoofed_forwarded_for_cannot_bypass_rate_limit():
    # Behind one trusted proxy, the proxy appends the real client IP as the LAST entry.
    client = _client(TRUSTED_PROXY_HOPS=1, RATE_LIMIT_SEARCH="2/minute")
    codes = [client.get("/api/search?q=spots", headers={"X-Forwarded-For": f"10.0.0.{i}, 203.0.113.7"}).status_code
             for i in range(3)]
    assert codes == [200, 200, 429]


def test_distinct_real_clients_have_separate_limits():
    client = _client(TRUSTED_PROXY_HOPS=1, RATE_LIMIT_SEARCH="2/minute")
    for ip in ("198.51.100.1", "198.51.100.2"):
        for _ in range(2):
            assert client.get("/api/search?q=spots", headers={"X-Forwarded-For": ip}).status_code == 200


def test_forwarded_for_ignored_without_trusted_proxy():
    client = _client(TRUSTED_PROXY_HOPS=0, RATE_LIMIT_SEARCH="2/minute")
    codes = [client.get("/api/search?q=spots", headers={"X-Forwarded-For": f"10.0.0.{i}"}).status_code for i in range(3)]
    assert codes == [200, 200, 429]


def test_client_ip_header_used_when_configured():
    client = _client(CLIENT_IP_HEADER="CF-Connecting-IP", RATE_LIMIT_SEARCH="2/minute")

    def get(ip, xff):
        return client.get("/api/search?q=spots", headers={"CF-Connecting-IP": ip, "X-Forwarded-For": xff}).status_code

    assert [get("198.51.100.9", f"10.0.0.{i}") for i in range(3)] == [200, 200, 429]
    assert get("198.51.100.10", "10.0.0.1") == 200
