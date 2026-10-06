"""CauliCare Similarity service: search engine (DINOv2 + FAISS) and its HTTP API."""
import io
import json

import numpy as np
import pytest
from PIL import Image

from similarity_service.app import create_app as create_service
from similarity_service.config import SERVICE_DIR, TestConfig
from similarity_service.engine import (
    EMBEDDING_DIM,
    ImageEncoder,
    SimilaritySearch,
    SimilarityUnavailableError,
    VectorIndex,
)
from tests.conftest import image_bytes, leafy_bytes

LABELS = ["Bacterial spot rot", "Black Rot", "Downy Mildew", "No disease"]
REAL_MODEL = SERVICE_DIR / "models" / "dinov2-small.onnx"
REAL_INDEX = SERVICE_DIR / "index"


# ------------------------------------------------------------------ helpers
def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


class StubEncoder:
    """Deterministic encoder: the embedding is the image's mean colour, padded to 384-d."""

    available = True
    _session = object()

    def load(self):
        return self

    def embed(self, img):
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
        return unit(np.concatenate([rgb, np.full(EMBEDDING_DIM - 3, 1.0)]))


def solid(color, size=(64, 64)):
    return Image.new("RGB", size, color)


def png(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def noisy_bytes(rgb, size=(300, 300), fmt="JPEG"):
    """A textured image around one colour (passes the main app's quality gate)."""
    rng = np.random.default_rng(3)
    arr = np.clip(np.array(rgb) + rng.normal(0, 14, (size[1], size[0], 3)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


def write_validator(path):
    """With the stub encoder (mean colour), this validator accepts green-dominated images."""
    w = np.zeros(EMBEDDING_DIM)
    w[:3] = (-40.0, 80.0, -40.0)
    (path / "leaf_validator.json").write_text(json.dumps(
        {"model": "stub", "version": "test", "weights": w.tolist(), "bias": 0.0, "threshold": 0.5}))


def make_index(path):
    """A tiny reference index: 4 solid-colour 'images', one per class, plus a stub validator."""
    enc = StubEncoder()
    colors = [(200, 40, 40), (40, 40, 40), (200, 200, 40), (40, 160, 60)]
    np.save(path / "embeddings.npy", np.stack([enc.embed(solid(c)) for c in colors]))
    items = [{"id": f"{label.lower().replace(' ', '-')}-001", "label": label, "split": "train",
              "source": f"{label}/x.jpg"} for label in LABELS]
    (path / "metadata.json").write_text(json.dumps({"info": {"count": 4}, "items": items}))
    (path / "thumbs").mkdir()
    for it, c in zip(items, colors):
        solid(c).save(path / "thumbs" / f"{it['id']}.jpg")
    write_validator(path)
    return path


def stub_search(index_dir, tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"stub")  # exists, but the stub encoder is used instead
    s = SimilaritySearch(model, index_dir)
    s.encoder = StubEncoder()
    s.load()
    return s


@pytest.fixture
def index_dir(tmp_path):
    d = tmp_path / "index"
    d.mkdir()
    return make_index(d)


@pytest.fixture
def search(index_dir, tmp_path):
    return stub_search(index_dir, tmp_path)


@pytest.fixture
def client(search):
    return create_service(TestConfig, search=search).test_client()


def post(client, data, name="leaf.png", **form):
    return client.post("/search", data={"image": (io.BytesIO(data), name), **form}, content_type="multipart/form-data")


# ------------------------------------------------------------------- engine
def test_vector_index_returns_exact_top_k():
    vecs = np.stack([unit(np.eye(EMBEDDING_DIM)[i] + 0.1) for i in range(6)])
    idx = VectorIndex(vecs)
    hits = idx.search(vecs[2], 3)
    assert hits[0][0] == 2 and abs(hits[0][1] - 1.0) < 1e-5
    assert [s for _, s in hits] == sorted([s for _, s in hits], reverse=True)
    assert len(idx.search(vecs[0], 50)) == 6  # k larger than the index is capped


def test_numpy_and_faiss_backends_agree():
    pytest.importorskip("faiss")  # blocked on some Windows hosts; always present in Docker / Render
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(50, EMBEDDING_DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    fa = VectorIndex(vecs)
    assert fa.backend == "faiss"
    np_idx = VectorIndex.__new__(VectorIndex)
    np_idx.vectors, np_idx._faiss, np_idx.backend = vecs, None, "numpy"
    q = vecs[7] + 0.05
    q /= np.linalg.norm(q)
    assert [i for i, _ in fa.search(q, 5)] == [i for i, _ in np_idx.search(q, 5)]


def test_index_loads_and_maps_metadata(search):
    assert search.available and len(search.index) == 4
    out = search.search(solid((200, 200, 40)), k=3)
    top = out["results"][0]
    assert top["label"] == "Downy Mildew" and top["id"] == "downy-mildew-001"
    assert top["thumbnail"].startswith("data:image/jpeg;base64,")
    assert len(out["results"]) == 3 and 0 <= top["similarity"] <= 1
    assert set(out["timing_ms"]) == {"embedding", "search"}
    assert "source" not in top  # dataset paths are never sent to clients


def test_missing_index_disables_feature(tmp_path):
    s = stub_search(tmp_path / "no-index", tmp_path)
    assert s.status == "unavailable" and not s.available
    with pytest.raises(SimilarityUnavailableError):
        s.search(solid((0, 0, 0)))


def test_missing_model_disables_feature(index_dir, tmp_path):
    s = SimilaritySearch(tmp_path / "missing.onnx", index_dir)
    s.load()
    assert s.status == "unavailable"


def test_mismatched_metadata_disables_feature(index_dir, tmp_path):
    meta = json.loads((index_dir / "metadata.json").read_text())
    meta["items"] = meta["items"][:2]
    (index_dir / "metadata.json").write_text(json.dumps(meta))
    assert stub_search(index_dir, tmp_path).status == "unavailable"


@pytest.mark.parametrize("bad_id", ["../metadata", "Black-Rot-001", "a/b", "", "x" * 65])
def test_unsafe_metadata_ids_are_rejected(index_dir, tmp_path, bad_id):
    meta = json.loads((index_dir / "metadata.json").read_text())
    meta["items"][0]["id"] = bad_id
    (index_dir / "metadata.json").write_text(json.dumps(meta))
    assert stub_search(index_dir, tmp_path).status == "unavailable"


def test_missing_thumbnail_disables_feature(index_dir, tmp_path):
    (index_dir / "thumbs" / "black-rot-001.jpg").unlink()
    assert stub_search(index_dir, tmp_path).status == "unavailable"


def test_thumbnail_lookup_only_by_indexed_id(search):
    assert search.thumbnail_data("black-rot-001").startswith("data:image/jpeg;base64,")
    for bad in ["../metadata", "black-rot-999", "/etc/passwd"]:
        with pytest.raises(KeyError):
            search.thumbnail_data(bad)


def test_encoder_preprocessing_matches_dinov2_config():
    enc = ImageEncoder(REAL_MODEL)
    x = enc.preprocess(solid((255, 0, 0), size=(640, 320)))
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    assert abs(float(x[0, 0].mean()) - (1 - 0.485) / 0.229) < 1e-4  # ImageNet normalisation


@pytest.mark.skipif(not REAL_MODEL.exists(), reason="DINOv2 model not downloaded (similarity_service/download_model.py)")
def test_real_encoder_embedding_is_normalised_and_deterministic():
    enc = ImageEncoder(REAL_MODEL)
    img = Image.open(io.BytesIO(leafy_bytes()))
    a, b = enc.embed(img), enc.embed(img)
    assert a.shape == (EMBEDDING_DIM,) and abs(float(np.linalg.norm(a)) - 1) < 1e-4
    assert np.allclose(a, b)
    assert float(a @ enc.embed(solid((20, 20, 200), size=(300, 300)))) < 0.99


@pytest.mark.skipif(not REAL_MODEL.exists(), reason="DINOv2 model not downloaded (similarity_service/download_model.py)")
def test_real_model_and_committed_index_load():
    s = SimilaritySearch(REAL_MODEL, REAL_INDEX)
    s.load()
    assert s.available and len(s.index) == 557
    out = s.search(Image.open(io.BytesIO(leafy_bytes())), k=5)
    assert len(out["results"]) == 5 and all(r["label"] in LABELS for r in out["results"])


def test_committed_index_is_consistent_and_attributed():
    meta = json.loads((REAL_INDEX / "metadata.json").read_text(encoding="utf-8"))
    vecs = np.load(REAL_INDEX / "embeddings.npy")
    assert vecs.shape == (len(meta["items"]), EMBEDDING_DIM) == (557, EMBEDDING_DIM)
    assert all((REAL_INDEX / "thumbs" / f"{it['id']}.jpg").is_file() for it in meta["items"])
    assert {it["split"] for it in meta["items"]} == {"train", "val"}  # test split never indexed
    attribution = (REAL_INDEX / "ATTRIBUTION.md").read_text(encoding="utf-8")
    assert "10.17632/t5sssfgn2v.3" in attribution and "CC BY 4.0" in attribution


# ---------------------------------------------------------------------- API
def test_health_reports_model_and_index(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json == {"status": "ok", "model_loaded": True, "index_loaded": True,
                      "reference_images": 4, "backend": r.json["backend"], "validator_loaded": True}


def test_health_is_503_when_unavailable(tmp_path):
    s = stub_search(tmp_path / "no-index", tmp_path)
    r = create_service(TestConfig, search=s).test_client().get("/health")
    assert r.status_code == 503 and r.json["status"] == "unavailable" and r.json["index_loaded"] is False
    r = create_service(TestConfig, search=s).test_client().post("/search")
    assert r.status_code == 503


def test_search_valid_image_returns_top_k(client):
    r = post(client, png(solid((200, 200, 40))), k="3")
    assert r.status_code == 200 and r.json["success"] is True
    res = r.json["results"]
    assert len(res) == 3 and res[0]["label"] == "Downy Mildew"
    for item in res:
        assert set(item) == {"id", "label", "similarity", "thumbnail"}
        assert 0 <= item["similarity"] <= 1 and item["thumbnail"].startswith("data:image/jpeg;base64,")
    assert [i["similarity"] for i in res] == sorted([i["similarity"] for i in res], reverse=True)


def test_k_is_clamped_and_validated(client):
    assert len(post(client, png(solid((1, 2, 3))), k="99").json["results"]) == 4  # capped by index size
    assert len(post(client, png(solid((1, 2, 3))), k="0").json["results"]) == 1
    assert post(client, png(solid((1, 2, 3))), k="abc").status_code == 400


@pytest.mark.parametrize("data,name", [
    (b"not an image at all", "x.jpg"),
    (b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, "truncated.png"),  # malformed PNG
    (b"", "empty.jpg"),
])
def test_invalid_or_malformed_images_are_rejected(client, data, name):
    r = post(client, data, name=name)
    assert r.status_code == 415 and r.json["code"] == "invalid_image"


def test_unsupported_format_is_rejected(client):
    buf = io.BytesIO()
    solid((10, 10, 10)).save(buf, format="GIF")
    assert post(client, buf.getvalue(), name="x.gif").status_code == 415


def test_missing_file_and_oversized_upload(client, search):
    assert client.post("/search", data={}).status_code == 400

    class Small(TestConfig):
        MAX_CONTENT_LENGTH = 1000

    small = create_service(Small, search=search).test_client()
    assert post(small, image_bytes(size=(200, 200))).status_code == 413


def test_token_is_required_when_configured(search):
    class Locked(TestConfig):
        SERVICE_TOKEN = "s3cret-token"

    c = create_service(Locked, search=search).test_client()
    data = png(solid((200, 200, 40)))
    assert post(c, data).status_code == 401
    r = c.post("/search", data={"image": (io.BytesIO(data), "a.png")}, headers={"X-Service-Token": "wrong"})
    assert r.status_code == 401
    r = c.post("/search", data={"image": (io.BytesIO(data), "a.png")}, headers={"X-Service-Token": "s3cret-token"})
    assert r.status_code == 200
    assert c.get("/health").status_code == 200  # health stays open for the platform


@pytest.mark.parametrize("path", ["/thumbs/black-rot-001.jpg", "/index/metadata.json", "/../app.py",
                                  "/static/../config.py", "/reference/black-rot-001.jpg"])
def test_no_file_serving_endpoints(client, path):
    assert client.get(path).status_code == 404


def test_errors_never_leak_internals(search, tmp_path):
    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    search.search = boom
    r = post(create_service(TestConfig, search=search).test_client(), png(solid((1, 1, 1))))
    assert r.status_code == 500 and "secret internal detail" not in r.get_data(as_text=True)


# ---------------------------------------------------------- leaf validator
def verify(client, data, name="leaf.jpg", **headers):
    return client.post("/verify", data={"image": (io.BytesIO(data), name)}, content_type="multipart/form-data",
                       headers=headers)


def test_verify_accepts_and_rejects(client):
    ok = verify(client, noisy_bytes((60, 140, 70)))
    assert ok.status_code == 200 and ok.json["is_cauliflower_leaf"] is True
    assert ok.json["score"] >= ok.json["threshold"]
    no = verify(client, noisy_bytes((190, 60, 50)))
    assert no.status_code == 200 and no.json["is_cauliflower_leaf"] is False
    assert no.json["score"] < no.json["threshold"]
    assert "results" not in no.json  # verification never returns similar images


def test_verify_then_search_reuses_the_embedding(search):
    calls = []
    embed = search.encoder.embed
    search.encoder.embed = lambda img: calls.append(1) or embed(img)
    c = create_service(TestConfig, search=search).test_client()
    data = noisy_bytes((60, 140, 70))
    assert verify(c, data).status_code == 200
    assert post(c, data).status_code == 200
    assert len(calls) == 1  # one DINOv2 pass for both calls
    post(c, noisy_bytes((61, 141, 71)))
    assert len(calls) == 2  # different bytes: embedded again


def test_verify_requires_token_and_valid_image(search):
    class Locked(TestConfig):
        SERVICE_TOKEN = "s3cret-token"

    c = create_service(Locked, search=search).test_client()
    token = {"X-Service-Token": "s3cret-token"}
    assert verify(c, noisy_bytes((60, 140, 70))).status_code == 401
    assert verify(c, noisy_bytes((60, 140, 70)), **token).status_code == 200
    assert verify(c, b"not an image", **token).status_code == 415
    assert c.post("/verify", data={}, headers=token).status_code == 400


def test_missing_or_bad_validator_makes_verify_503_but_search_works(index_dir, tmp_path):
    (index_dir / "leaf_validator.json").unlink()
    s = stub_search(index_dir, tmp_path)
    c = create_service(TestConfig, search=s).test_client()
    assert verify(c, noisy_bytes((60, 140, 70))).status_code == 503
    assert post(c, noisy_bytes((60, 140, 70))).status_code == 200
    assert c.get("/health").json["validator_loaded"] is False
    for bad in ({"weights": [1.0] * 10, "bias": 0, "threshold": 0.5},
                {"weights": [0.0] * EMBEDDING_DIM, "bias": 0, "threshold": 1.5},
                {"weights": [float("nan")] * EMBEDDING_DIM, "bias": 0, "threshold": 0.5}):
        (index_dir / "leaf_validator.json").write_text(json.dumps(bad))
        assert stub_search(index_dir, tmp_path).validator_available is False


def test_committed_validator_is_well_formed():
    from similarity_service.validator import LeafValidator

    v = LeafValidator(REAL_INDEX / "leaf_validator.json").load()
    spec = json.loads((REAL_INDEX / "leaf_validator.json").read_text())
    assert v.weights.shape == (EMBEDDING_DIM,) and 0 < v.threshold < 1
    assert spec["report"] == "reports/leaf_validator.json"


@pytest.mark.skipif(not REAL_MODEL.exists(), reason="DINOv2 model not downloaded")
def test_real_validator_separates_dataset_leaves_from_unrelated_images():
    """Real DINOv2 + committed validator: reference thumbnails (cauliflower leaves) are
    accepted; synthetic non-leaf images (a text document, flat-colour shapes) are rejected."""
    from PIL import ImageDraw

    s = SimilaritySearch(REAL_MODEL, REAL_INDEX)
    s.load()
    assert s.validator_available
    thumbs = sorted((REAL_INDEX / "thumbs").glob("*.jpg"))[::40]
    accepted = [s.verify(Image.open(t).convert("RGB"))["is_cauliflower_leaf"] for t in thumbs]
    assert sum(accepted) >= len(accepted) - 1, accepted
    doc = Image.new("RGB", (800, 1000), "white")
    d = ImageDraw.Draw(doc)
    for y in range(60, 960, 30):
        d.text((50, y), "Quarterly revenue summary and payment table " * 2, fill="black")
    shapes = Image.new("RGB", (640, 480), (30, 60, 160))
    ImageDraw.Draw(shapes).ellipse((100, 80, 400, 380), fill=(240, 200, 30))
    for img in (doc, shapes):
        assert s.verify(img)["is_cauliflower_leaf"] is False
