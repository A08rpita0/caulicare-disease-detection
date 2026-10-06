"""JSON API: image analysis, disease knowledge, health."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from flask import Blueprint, current_app, jsonify, request

from services.analysis import (
    DISCLAIMER,
    LEAF_CHECK_UNAVAILABLE,
    build_result,
    not_cauliflower_outcome,
    suitability_outcome,
)
from services.image_validation import (
    ImageValidationError,
    allowed_extension,
    allowed_mimetype,
    assess_quality,
    load_image,
)
from services.predictor import (
    CLASS_NAMES,
    CONFIDENCE_TIERS,
    HIGH_SCORE_THRESHOLD,
    LOW_SCORE_THRESHOLD,
    ModelUnavailableError,
)
from services.security import rate_limited
from services.severity import SeverityError
from services.severity import estimate as estimate_severity
from services.similarity_client import UNAVAILABLE

log = logging.getLogger(__name__)
bp = Blueprint("api", __name__, url_prefix="/api")

UNSUPPORTED = "Please upload a JPG, PNG, or WEBP image."
# Similarity requests wait on the network, so they run alongside the CNN.
_similarity_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="similarity")


def error(message: str, status: int, code: str):
    return jsonify(success=False, error=message, code=code), status


def services():
    return current_app.extensions["caulicare"]


class Refused(Exception):
    """Ends the request with a ready-made JSON response."""

    def __init__(self, response):
        super().__init__()
        self.response = response


def _read_upload():
    """Bytes of the uploaded image after the request-level checks."""
    cfg = current_app.config
    file = request.files.get("image")
    if file is None or not file.filename:
        raise Refused(error("No image was uploaded. Please choose a photo of a cauliflower leaf.", 400, "no_file"))
    if not allowed_extension(file.filename, cfg["ALLOWED_EXTENSIONS"]):
        raise Refused(error(UNSUPPORTED, 415, "unsupported_extension"))
    if not allowed_mimetype(file.mimetype):
        raise Refused(error(UNSUPPORTED, 415, "unsupported_mimetype"))
    data = file.read(cfg["MAX_CONTENT_LENGTH"] + 1)  # processed in memory; never written to disk
    if len(data) > cfg["MAX_CONTENT_LENGTH"]:
        raise Refused(error(f"The image is too large. Please upload an image under {cfg['MAX_UPLOAD_MB']:g} MB.",
                            413, "file_too_large"))
    return file, data


def _decode(data: bytes):
    cfg = current_app.config
    try:
        return load_image(data, cfg["ALLOWED_FORMATS"], cfg["MAX_IMAGE_PIXELS"])
    except ImageValidationError as exc:
        raise Refused(error(str(exc), exc.status, exc.code)) from exc


UNCHECKABLE = {"blank_image", "low_resolution"}


def _screen(file, data: bytes):
    """Image checks, then the cauliflower-leaf check. Returns (quality, leaf_check) for an
    image the disease model may analyse; raises Refused for anything else.

    Order: file decoding -> quality gate -> cauliflower-leaf validator (blank and tiny
    images are refused before it; for dark, overexposed or blurry images its verdict
    decides which refusal the user sees). The validator runs
    in the similarity service (it needs the DINOv2 embedding); if that check cannot be
    made, the upload is refused (fail closed) rather than analysed unverified."""
    predictor = services()["predictor"]
    with predictor.slot():  # decoding is memory-heavy: one at a time, like inference
        img = _decode(data)
        quality = assess_quality(img)
        del img
    refusal = suitability_outcome(quality)

    def refuse_quality():
        raise Refused((jsonify(success=False, code=refusal["status"], quality=quality.to_dict(),
                               disclaimer=DISCLAIMER, **refusal), 422))

    # Blank or tiny images carry nothing to check: refused straight away.
    if refusal and {p.code for p in quality.problems} & UNCHECKABLE:
        refuse_quality()
    # Outside the slot: waiting for a waking service must not block other users' analyses.
    leaf = services()["similarity"].verify(data, file.filename, file.mimetype)
    # A dark, overexposed or blurry image that is clearly not a cauliflower leaf (a white
    # document, a screenshot) is told so, rather than given photo-quality advice.
    if refusal and not (leaf.get("checked") and not leaf["is_cauliflower_leaf"]):
        refuse_quality()
    if not leaf.get("checked"):
        raise Refused(error(LEAF_CHECK_UNAVAILABLE, 503, "leaf_check_unavailable"))
    if not leaf["is_cauliflower_leaf"]:
        out = not_cauliflower_outcome(leaf)
        raise Refused((jsonify(success=False, code=out["status"], quality=quality.to_dict(),
                               disclaimer=DISCLAIMER, **out), 422))
    return quality, leaf


@bp.post("/verify")
@rate_limited("RATE_LIMIT_PREDICT")
def verify():
    """Only the image checks and the cauliflower-leaf check. The Analyze page calls this
    first so it can show each step; /api/predict repeats the same checks itself."""
    try:
        file, data = _read_upload()
        quality, leaf = _screen(file, data)
    except Refused as r:
        return r.response
    return jsonify(success=True, status="cauliflower_leaf", leaf_check=leaf, quality=quality.to_dict())


@bp.post("/predict")
@rate_limited("RATE_LIMIT_PREDICT")
def predict():
    explain = request.args.get("explain", "1") != "0"
    predictor = services()["predictor"]
    kb = services()["kb"]
    try:
        file, data = _read_upload()
        # The disease model only runs on images that pass every check, including
        # the cauliflower-leaf validator, whatever the client did before.
        quality, leaf = _screen(file, data)
        with predictor.slot():
            img = _decode(data)
            try:
                sim = services()["similarity"]
                similar_job = _similarity_pool.submit(sim.search, data, file.filename, file.mimetype)
                prediction = predictor.predict(img, explain=explain)
            except ModelUnavailableError:
                return error("The disease detection model is not available on this server yet.", 503,
                             "model_unavailable")
            except Exception:  # noqa: BLE001 - never expose internals
                log.exception("Analysis failed")
                return error("We couldn't process this image. Please try another photo.", 500, "analysis_failed")
            # The two additional analyses are independent of the classifier: a failure
            # in either is reported in its own section and never breaks the prediction.
            severity_result = _severity(img)
            del img
            similar = _similar(similar_job, sim.timeout)
    except Refused as r:
        return r.response

    result = build_result(prediction, quality, kb)
    return jsonify(success=True, quality=quality.to_dict(), leaf_check=leaf, severity=severity_result,
                   similar_images=similar, **result)


def _severity(img) -> dict:
    try:
        return estimate_severity(img).to_dict()
    except SeverityError as exc:
        return {"available": False, "reason": f"Not estimated: {exc}."}
    except Exception:  # noqa: BLE001
        log.exception("Affected-area estimate failed")
        return {"available": False, "reason": "The affected-area estimate could not be produced for this image."}


def _similar(job, timeout: float) -> dict:
    # The client never raises; the extra margin only guards against a hung thread.
    try:
        return job.result(timeout=timeout + 5)
    except Exception:  # noqa: BLE001
        log.warning("Similarity search did not finish in time")
        return {"available": False, "reason": UNAVAILABLE}


@bp.get("/diseases")
def diseases():
    kb = services()["kb"]
    return jsonify(success=True, diseases=[kb.public_record(d) for d in kb.diseases], disclaimer=kb.meta["disclaimer"])


@bp.get("/diseases/<name>")
def disease(name: str):
    kb = services()["kb"]
    record = kb.get(name)
    if not record:
        return error("Disease not found.", 404, "not_found")
    return jsonify(success=True, disease=kb.public_record(record))


@bp.get("/search")
@rate_limited("RATE_LIMIT_SEARCH")
def search():
    q = (request.args.get("q") or "").strip()[:200]
    if not q:
        return error("Please enter a symptom or keyword.", 400, "empty_query")
    return jsonify(success=True, query=q, results=services()["kb"].search(q))


@bp.get("/health")
def health():
    p = services()["predictor"]
    return jsonify(
        success=True,
        status="ok",
        model_loaded=p.is_loaded,
        model_file_present=p.model_path.exists(),
        classes=CLASS_NAMES,
        confidence_tiers={
            "uncertain": f"< {LOW_SCORE_THRESHOLD:.2f}",
            "moderate": f"{LOW_SCORE_THRESHOLD:.2f} - {HIGH_SCORE_THRESHOLD:.2f}",
            "high": f">= {HIGH_SCORE_THRESHOLD:.2f}",
        },
        tier_labels={k: v[0] for k, v in CONFIDENCE_TIERS.items()},
        # Never calls the other service: this endpoint is the platform's health check.
        similarity_search=services()["similarity"].configured,
        # Analyses need the cauliflower-leaf check, which runs in the similarity service.
        leaf_validation=services()["similarity"].configured,
        similarity_service=services()["similarity"].last_status,
    )
