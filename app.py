"""CauliCare - Flask application factory."""
from __future__ import annotations

import logging
import mimetypes
import os
import uuid

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix

from config import BASE_DIR, Config

# Some hosts (notably Windows) map .js to text/plain, which browsers refuse to
# execute under `X-Content-Type-Options: nosniff`. Pin the correct types.
for _ext, _type in {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".webp": "image/webp"}.items():
    mimetypes.add_type(_type, _ext)
from services.cache import ResultCache
from services.knowledge_base import KnowledgeBase
from services.predictor import CLASS_NAMES, ModelUnavailableError, Predictor
from services.security import apply_security_headers
from services.similarity_client import SimilarityClient
from services.telemetry import current_request_id, init_telemetry


def create_app(config_object=Config, predictor: Predictor | None = None,
               similarity: SimilarityClient | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_object)
    hops = app.config.get("TRUSTED_PROXY_HOPS", 0)
    if hops:
        # Trust only the right-most `hops` X-Forwarded-* entries (those appended by our
        # own proxies); anything to their left is client-controlled and ignored.
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=hops, x_proto=hops, x_host=0, x_prefix=0)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    kb = KnowledgeBase(BASE_DIR / "data" / "diseases.json")
    predictor = predictor or Predictor(app.config["MODEL_PATH"], app.config["MODEL_URL"],
                                       app.config["MAX_CONCURRENT_PREDICTIONS"],
                                       onnx_path=app.config.get("ONNX_MODEL_PATH"),
                                       onnx_threads=app.config.get("ONNX_THREADS", 1),
                                       backend=app.config.get("INFERENCE_BACKEND", "auto"))
    similarity = similarity or SimilarityClient(
        app.config["SIMILARITY_SERVICE_URL"], app.config["SIMILARITY_TIMEOUT"],
        app.config["SIMILARITY_SERVICE_TOKEN"], app.config["SIMILAR_RESULTS"], CLASS_NAMES,
        app.config["LEAF_CHECK_TIMEOUT"])
    cache = ResultCache(app.config.get("PREDICTION_CACHE_SIZE", 0))
    app.extensions["caulicare"] = {"kb": kb, "predictor": predictor, "similarity": similarity, "cache": cache}
    init_telemetry(app)

    app.extensions["rate_ns"] = uuid.uuid4().hex  # isolates rate-limit buckets per app instance

    if app.config["PRELOAD_MODEL"]:
        try:
            predictor.load()
        except ModelUnavailableError:
            app.logger.warning("Starting without a model; /api/predict will return 503 until it is installed.")
        similarity.warm_up(0)  # wake the similarity service along with this one

    from routes.api import bp as api_bp
    from routes.pages import bp as pages_bp

    app.register_blueprint(pages_bp)
    app.register_blueprint(api_bp)
    app.after_request(apply_security_headers)

    @app.context_processor
    def inject_globals():
        return {"max_upload_mb": app.config["MAX_UPLOAD_MB"]}

    def wants_json():
        return request.path.startswith("/api/")

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        message = f"The image is too large. Please upload an image under {app.config['MAX_UPLOAD_MB']:g} MB."
        if wants_json():
            return jsonify(success=False, error=message, code="file_too_large", decision="rejected",
                           request_id=current_request_id()), 413
        return render_template("error.html", code=413, message=message), 413

    @app.errorhandler(HTTPException)
    def http_error(exc):
        if wants_json():
            return jsonify(success=False, error=exc.description, code=exc.name.lower().replace(" ", "_"),
                           request_id=current_request_id()), exc.code
        return render_template("error.html", code=exc.code, message=exc.description), exc.code

    @app.errorhandler(Exception)
    def unhandled(exc):  # never leak stack traces to clients
        app.logger.exception("Unhandled error: %s", exc)
        if wants_json():
            return jsonify(success=False, error="Something went wrong on our side. Please try again.",
                           code="internal_error", decision="error", request_id=current_request_id()), 500
        return render_template("error.html", code=500, message="Something went wrong on our side."), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=app.config["DEBUG"])
