"""Main app -> similarity service: client validation, failure handling and the integrated flow.

The servers here are real HTTP servers on localhost (werkzeug), so the client's
networking, timeouts and parsing are exercised for real.
"""
import io
import json
import threading
import time

import pytest
from flask import Flask
from werkzeug.serving import make_server

from app import create_app
from config import TestConfig
from services.predictor import CLASS_NAMES
from services.similarity_client import NOT_CONFIGURED, UNAVAILABLE, SimilarityClient
from similarity_service.app import create_app as create_service
from similarity_service.config import TestConfig as ServiceTestConfig
from tests.conftest import leafy_bytes
from tests.test_similarity_service import make_index, noisy_bytes, stub_search

THUMB = "data:image/jpeg;base64,/9j/AAAA"


class Server:
    def __init__(self, app):
        self.srv = make_server("127.0.0.1", 0, app, threaded=True)
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.srv.shutdown()


LEAF_OK = {"success": True, "is_cauliflower_leaf": True, "score": 0.97, "threshold": 0.5, "model": "x"}


def fake_service(payload=None, status=200, delay=0.0, raw=None, verdict=None, verify_delay=0.0):
    app = Flask("fake-similarity")

    @app.post("/verify")
    def verify():
        time.sleep(verify_delay)
        return json.dumps(verdict or LEAF_OK), 200, {"Content-Type": "application/json"}

    @app.post("/search")
    def search():
        time.sleep(delay)
        if raw is not None:
            return raw, status, {"Content-Type": "application/json"}
        return json.dumps(payload), status, {"Content-Type": "application/json"}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


def good_payload(n=3):
    return {"success": True, "model": "DINOv2-small (ONNX)", "backend": "faiss", "reference_images": 557,
            "timing_ms": {"embedding": 12.0, "search": 0.1},
            "results": [{"id": f"black-rot-00{i}", "label": "Black Rot", "similarity": 0.9 - i / 10,
                         "thumbnail": THUMB} for i in range(1, n + 1)]}


def client_for(url, **kw):
    return SimilarityClient(url, labels=CLASS_NAMES, **{"timeout": 5, **kw})


# -------------------------------------------------------------- client unit
def test_not_configured():
    assert SimilarityClient("").search(b"x") == {"available": False, "reason": NOT_CONFIGURED}


def test_valid_response_is_passed_through():
    with Server(fake_service(good_payload())) as s:
        out = client_for(s.url).search(leafy_bytes(), "leaf.jpg", "image/jpeg")
    assert out["available"] is True and len(out["results"]) == 3
    assert out["results"][0] == {"id": "black-rot-001", "label": "Black Rot", "similarity": 0.8, "thumbnail": THUMB}
    assert out["backend"] == "faiss" and "service_round_trip" in out["timing_ms"]


def test_timeout_returns_unavailable_quickly():
    with Server(fake_service(good_payload(), delay=3)) as s:
        t0 = time.perf_counter()
        out = client_for(s.url, timeout=0.5).search(leafy_bytes())
        assert time.perf_counter() - t0 < 2.5
    assert out == {"available": False, "reason": UNAVAILABLE}


def test_service_down_returns_unavailable():
    srv = Server(fake_service(good_payload()))
    url = srv.url
    srv.srv.server_close()  # nothing listening on this port any more
    out = client_for(url).search(leafy_bytes())
    assert out == {"available": False, "reason": UNAVAILABLE}


@pytest.mark.parametrize("status", [401, 500, 503])
def test_error_status_returns_unavailable(status):
    with Server(fake_service({"success": False, "error": "x"}, status=status)) as s:
        assert client_for(s.url).search(leafy_bytes())["available"] is False


def bad(**changes):
    p = good_payload(1)
    p["results"][0].update(changes)
    return p


@pytest.mark.parametrize("payload", [
    bad(id="../../etc/passwd"),
    bad(label="Not a class"),
    bad(similarity=1.5),
    bad(similarity="0.9"),
    bad(thumbnail="https://evil.example/x.jpg"),
    bad(thumbnail="javascript:alert(1)"),
    bad(thumbnail='data:image/jpeg;base64,AAA"onerror="x'),
    {"success": True, "results": "nope"},
    {"success": False, "results": []},
    good_payload(9),  # more results than requested (k=5)
    ["not", "a", "dict"],
])
def test_invalid_responses_are_rejected(payload):
    with Server(fake_service(payload)) as s:
        assert client_for(s.url).search(leafy_bytes()) == {"available": False, "reason": UNAVAILABLE}


def test_non_json_response_is_rejected():
    with Server(fake_service(raw="<html>Service waking up</html>")) as s:
        assert client_for(s.url).search(leafy_bytes())["available"] is False


def test_extra_fields_are_dropped():
    p = good_payload(1)
    p["results"][0]["source"] = "Black Rot/secret.jpg"
    p["debug"] = "internal"
    with Server(fake_service(p)) as s:
        out = client_for(s.url).search(leafy_bytes())
    assert "source" not in out["results"][0] and "debug" not in out


def test_token_is_sent():
    seen = {}
    app = Flask("token-check")

    @app.post("/search")
    def search():
        from flask import request
        seen["token"] = request.headers.get("X-Service-Token")
        return good_payload(1)

    with Server(app) as s:
        client_for(s.url, token="abc123").search(leafy_bytes())
    assert seen["token"] == "abc123"


def test_warm_up_is_throttled_and_records_status():
    with Server(fake_service(good_payload())) as s:
        c = client_for(s.url)
        c.warm_up(min_interval=60)
        c.warm_up(min_interval=60)  # throttled: no second thread
        for _ in range(50):
            if c.last_status["reachable"] is not None:
                break
            time.sleep(0.05)
    assert c.last_status["reachable"] is True


# ---------------------------------------------------------- leaf check unit
def test_verify_not_configured_is_unchecked():
    assert SimilarityClient("").verify(b"x") == {"checked": False}


def test_verify_valid_answers():
    with Server(fake_service(good_payload())) as s:
        assert client_for(s.url).verify(leafy_bytes()) == {
            "checked": True, "is_cauliflower_leaf": True, "score": 0.97, "threshold": 0.5}
    no = {"success": True, "is_cauliflower_leaf": False, "score": 0.03, "threshold": 0.5}
    with Server(fake_service(good_payload(), verdict=no)) as s:
        assert client_for(s.url).verify(leafy_bytes())["is_cauliflower_leaf"] is False


@pytest.mark.parametrize("verdict", [
    {"success": True, "is_cauliflower_leaf": True, "score": 0.1, "threshold": 0.5},  # contradicts its score
    {"success": True, "is_cauliflower_leaf": "yes", "score": 0.9, "threshold": 0.5},
    {"success": True, "is_cauliflower_leaf": True, "score": 1.7, "threshold": 0.5},
    {"success": True, "is_cauliflower_leaf": True, "score": 0.9},
    {"success": False, "is_cauliflower_leaf": True, "score": 0.9, "threshold": 0.5},
])
def test_verify_malformed_answers_are_unchecked(verdict):
    with Server(fake_service(good_payload(), verdict=verdict)) as s:
        assert client_for(s.url).verify(leafy_bytes()) == {"checked": False}


def test_verify_timeout_and_service_down_are_unchecked():
    with Server(fake_service(good_payload(), verify_delay=3)) as s:
        assert client_for(s.url, verify_timeout=0.5).verify(leafy_bytes()) == {"checked": False}
    srv = Server(fake_service(good_payload()))
    srv.srv.server_close()
    assert client_for(srv.url).verify(leafy_bytes()) == {"checked": False}


def test_verify_sends_token_and_no_k():
    seen = {}
    app = Flask("verify-check")

    @app.post("/verify")
    def verify():
        from flask import request
        seen["token"], seen["form"] = request.headers.get("X-Service-Token"), dict(request.form)
        return LEAF_OK

    with Server(app) as s:
        client_for(s.url, token="abc123").verify(leafy_bytes())
    assert seen == {"token": "abc123", "form": {}}


# ----------------------------------------------------- main app integration
def upload(client, data, name="leaf.jpg"):
    return client.post("/api/predict", data={"image": (io.BytesIO(data), name)}, content_type="multipart/form-data")


def test_prediction_survives_similarity_timeout(loaded_predictor):
    with Server(fake_service(good_payload(), delay=3)) as s:
        main = create_app(TestConfig, predictor=loaded_predictor, similarity=client_for(s.url, timeout=0.5))
        body = upload(main.test_client(), leafy_bytes()).json
    assert body["status"] == "ok" and body["prediction"]["disease"] in CLASS_NAMES
    assert body["prediction"]["explanation"]["overlay"].startswith("data:image/jpeg;base64,")
    assert body["severity"]["available"] in (True, False)
    assert body["similar_images"] == {"available": False, "reason": UNAVAILABLE}


def test_prediction_survives_invalid_similarity_response(loaded_predictor):
    with Server(fake_service(raw="{broken json")) as s:
        main = create_app(TestConfig, predictor=loaded_predictor, similarity=client_for(s.url))
        r = upload(main.test_client(), leafy_bytes())
    assert r.status_code == 200 and r.json["status"] == "ok"
    assert r.json["similar_images"]["available"] is False
    assert "Traceback" not in r.get_data(as_text=True)


def test_main_to_similarity_service_end_to_end(loaded_predictor, tmp_path):
    """Real main app -> real HTTP -> real similarity-service app (stub encoder, tiny index)."""
    index = tmp_path / "index"
    index.mkdir()
    search = stub_search(make_index(index), tmp_path)

    class Locked(ServiceTestConfig):
        SERVICE_TOKEN = "shared-secret"

    with Server(create_service(Locked, search=search)) as s:
        main = create_app(TestConfig, predictor=loaded_predictor,
                          similarity=client_for(s.url, token="shared-secret"))
        body = upload(main.test_client(), leafy_bytes()).json
        wrong = create_app(TestConfig, predictor=loaded_predictor, similarity=client_for(s.url, token="wrong"))
        denied = upload(wrong.test_client(), leafy_bytes()).json
    sim = body["similar_images"]
    assert body["status"] == "ok" and sim["available"] is True
    assert len(sim["results"]) == 4 and {r["label"] for r in sim["results"]} == set(CLASS_NAMES)
    assert all(r["thumbnail"].startswith("data:image/jpeg;base64,") for r in sim["results"])
    assert body["leaf_check"]["is_cauliflower_leaf"] is True
    # Without the right token the leaf check cannot be made, so nothing is analysed.
    assert denied["code"] == "leaf_check_unavailable" and "prediction" not in denied


def test_leaf_check_timeout_refuses_quickly_without_running_the_model(loaded_predictor):
    """A hung similarity service must not hang the upload or let the disease model run."""
    with Server(fake_service(good_payload(), verify_delay=3)) as s:
        main = create_app(TestConfig, predictor=loaded_predictor, similarity=client_for(s.url, verify_timeout=0.5))
        t0 = time.perf_counter()
        r = upload(main.test_client(), leafy_bytes())
        elapsed = time.perf_counter() - t0
    assert r.status_code == 503 and r.json["code"] == "leaf_check_unavailable"
    assert "prediction" not in r.json and elapsed < 2.5


def test_main_to_service_end_to_end_rejection(loaded_predictor, tmp_path):
    """The real service app's validator rejects the image: the main app refuses it."""
    index = tmp_path / "index"
    index.mkdir()
    search = stub_search(make_index(index), tmp_path)
    with Server(create_service(ServiceTestConfig, search=search)) as s:
        main = create_app(TestConfig, predictor=loaded_predictor, similarity=client_for(s.url))
        r = upload(main.test_client(), noisy_bytes((190, 60, 50)), name="red.jpg")
    assert r.status_code == 422 and r.json["status"] == "not_cauliflower_leaf"
    assert "prediction" not in r.json and "similar_images" not in r.json
