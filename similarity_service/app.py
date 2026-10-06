"""CauliCare Similarity - HTTP API around the DINOv2 + FAISS search engine.

    GET  /health   model, index and validator status (200 when ready, 503 otherwise)
    POST /verify   multipart field `image` -> is it a cauliflower leaf? (score, threshold)
    POST /search   multipart field `image` (+ optional `k`) -> top-K similar reference images

Only the main CauliCare app is meant to call this service. Uploads are
validated again here (size, decoded format, pixel count): nothing from the
caller is trusted. Results only ever contain indexed reference images; no
endpoint takes a file path or an id.
"""
from __future__ import annotations

import hmac
import io
import logging
import threading

from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from .config import Config
from .engine import SimilaritySearch

log = logging.getLogger(__name__)


def error(message: str, status: int, code: str):
    return jsonify(success=False, error=message, code=code), status


def decode_image(data: bytes, cfg) -> Image.Image:
    if not data:
        raise ValueError("empty upload")
    Image.MAX_IMAGE_PIXELS = cfg["MAX_IMAGE_PIXELS"]
    with Image.open(io.BytesIO(data)) as im:
        if im.format not in cfg["ALLOWED_FORMATS"]:
            raise ValueError("unsupported format")
        if im.width * im.height > cfg["MAX_IMAGE_PIXELS"]:
            raise ValueError("image too large")
        im.load()
        return im.convert("RGB")


def create_app(config_object=Config, search: SimilaritySearch | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_object)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    search = search or SimilaritySearch(app.config["EMBEDDING_MODEL_PATH"], app.config["IMAGE_INDEX_DIR"])
    if app.config["PRELOAD"] and search.status == "not_loaded":
        search.load()  # model + index loaded once, before the first request
    slots = threading.BoundedSemaphore(max(1, app.config["MAX_CONCURRENT_SEARCHES"]))
    app.extensions["similarity"] = search

    @app.after_request
    def headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/health")
    def health():
        ready = search.available
        body = {
            "status": "ok" if ready else "unavailable",
            "model_loaded": ready and search.encoder._session is not None,
            "index_loaded": ready and search.index is not None,
            "reference_images": len(search.index) if ready else 0,
            "backend": search.index.backend if ready else None,
            "validator_loaded": search.validator_available,
        }
        return jsonify(body), 200 if ready else 503

    def authorised() -> bool:
        token = app.config["SERVICE_TOKEN"]
        return not token or hmac.compare_digest(request.headers.get("X-Service-Token", ""), token)

    def read_upload():
        """(image, bytes) from the multipart field `image`, or an error response."""
        file = request.files.get("image")
        if file is None:
            return None, error("No image was uploaded.", 400, "no_file")
        data = file.read(app.config["MAX_CONTENT_LENGTH"] + 1)
        if len(data) > app.config["MAX_CONTENT_LENGTH"]:
            return None, error("The image is too large.", 413, "file_too_large")
        try:
            return (decode_image(data, app.config), data), None
        except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError):
            return None, error("The file is not a valid JPG, PNG or WEBP image.", 415, "invalid_image")

    @app.post("/verify")
    def do_verify():
        if not authorised():
            return error("Unauthorized.", 401, "unauthorized")
        if not search.validator_available:
            return error("The leaf check is not available.", 503, "unavailable")
        upload, err = read_upload()
        if err:
            return err
        img, data = upload
        with slots:
            out = search.verify(img, data)
        return jsonify(success=True, **out)

    @app.post("/search")
    def do_search():
        if not authorised():
            return error("Unauthorized.", 401, "unauthorized")
        if not search.available:
            return error("Similarity search is not available.", 503, "unavailable")
        try:
            k = int(request.form.get("k", app.config["DEFAULT_K"]))
        except ValueError:
            return error("k must be an integer.", 400, "bad_k")
        k = max(1, min(k, app.config["MAX_K"]))
        upload, err = read_upload()
        if err:
            return err
        img, data = upload
        with slots:
            out = search.search(img, k, data)
        return jsonify(success=True, **out)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        return error("The image is too large.", 413, "file_too_large")

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return error(exc.description, exc.code, exc.name.lower().replace(" ", "_"))

    @app.errorhandler(Exception)
    def unhandled(exc):  # never leak internals
        log.exception("Unhandled error: %s", exc)
        return error("Internal error.", 500, "internal_error")

    return app


app = create_app()

if __name__ == "__main__":
    import os

    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8001")))
