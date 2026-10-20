"""JSON API: image analysis (single and batch), model information, service statistics, health.

Every JSON response carries `request_id` (also sent as the `X-Request-ID` header). Analysis
responses carry `decision`: "accepted", "uncertain" or "rejected" (see docs/API.md), and
`timing_ms`, the wall-clock time spent in each stage of the request.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from flask import Blueprint, current_app, jsonify, request, send_from_directory

from config import BASE_DIR
from services.analysis import (
    DEFAULT_TOP_K,
    DISCLAIMER,
    LEAF_CHECK_UNAVAILABLE,
    apply_second_opinion,
    build_result,
    compact_result,
    not_cauliflower_outcome,
    second_opinion,
    suitability_outcome,
)
from services.cache import fingerprint
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
from services.security import consume_rate_limit, rate_limited
from services.severity import SeverityError
from services.severity import estimate as estimate_severity
from services.similarity_client import NOT_CONFIGURED, UNAVAILABLE
from services.telemetry import Stopwatch, annotate, current_request_id

log = logging.getLogger(__name__)
bp = Blueprint("api", __name__, url_prefix="/api")

UNSUPPORTED = "Please upload a JPG, PNG, or WEBP image."
MULTIPART_OVERHEAD = 64 * 1024  # boundaries and headers around one file in a request
# Similarity requests wait on the network, so they run alongside the CNN.
_similarity_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="similarity")


# ------------------------------------------------------------------ responses
def err_payload(message: str, code: str, status: int) -> dict:
    # "rejected": the input was refused. "error": the service could not do its job.
    decision = "rejected" if status < 500 and status != 429 else "error"
    return {"success": False, "error": message, "code": code, "decision": decision,
            "request_id": current_request_id()}


def error(message: str, status: int, code: str):
    annotate(decision=err_payload(message, code, status)["decision"], error_code=code)
    return jsonify(err_payload(message, code, status)), status


def services():
    return current_app.extensions["caulicare"]


class Refused(Exception):
    """Ends the request with a ready-made JSON body and status."""

    def __init__(self, payload: dict, status: int):
        super().__init__(payload.get("error") or payload.get("code"))
        self.payload = payload
        self.status = status

    def response(self):
        annotate(decision=self.payload.get("decision"), error_code=self.payload.get("code"))
        return jsonify({**self.payload, "request_id": current_request_id()}), self.status


def refuse(message: str, status: int, code: str):
    raise Refused(err_payload(message, code, status), status)


# ------------------------------------------------------------------ uploads
def _top_k() -> int:
    raw = request.args.get("top_k")
    if raw is None:
        return DEFAULT_TOP_K
    if not raw.isdigit() or not 1 <= int(raw) <= len(CLASS_NAMES):
        refuse(f"top_k must be a whole number from 1 to {len(CLASS_NAMES)}.", 400, "invalid_parameter")
    return int(raw)


def _read_file(file):
    """Bytes of one uploaded file after the request-level checks."""
    cfg = current_app.config
    if file is None or not file.filename:
        refuse("No image was uploaded. Please choose a photo of a cauliflower leaf.", 400, "no_file")
    if not allowed_extension(file.filename, cfg["ALLOWED_EXTENSIONS"]):
        refuse(UNSUPPORTED, 415, "unsupported_extension")
    if not allowed_mimetype(file.mimetype):
        refuse(UNSUPPORTED, 415, "unsupported_mimetype")
    data = file.read(cfg["MAX_UPLOAD_BYTES"] + 1)  # processed in memory; never written to disk
    if len(data) > cfg["MAX_UPLOAD_BYTES"]:
        refuse(f"The image is too large. Please upload an image under {cfg['MAX_UPLOAD_MB']:g} MB.",
               413, "file_too_large")
    return data


def _read_upload():
    """The single `image` field of /predict and /verify: (file, bytes)."""
    cfg = current_app.config
    # The framework limit leaves room for a whole batch, so hold single uploads to one image here.
    if request.content_length and request.content_length > cfg["MAX_UPLOAD_BYTES"] + MULTIPART_OVERHEAD:
        refuse(f"The image is too large. Please upload an image under {cfg['MAX_UPLOAD_MB']:g} MB.",
               413, "file_too_large")
    file = request.files.get("image")
    return file, _read_file(file)


def _decode(data: bytes):
    cfg = current_app.config
    try:
        return load_image(data, cfg["ALLOWED_FORMATS"], cfg["MAX_IMAGE_PIXELS"])
    except ImageValidationError as exc:
        refuse(str(exc), exc.status, exc.code)


UNCHECKABLE = {"blank_image", "low_resolution"}


def _refusal(code: str, quality, outcome: dict) -> Refused:
    payload = {"success": False, "code": code, "quality": quality.to_dict(), "disclaimer": DISCLAIMER,
               "decision": "rejected", **outcome}
    return Refused(payload, 422)


def _screen(file, data: bytes, sw: Stopwatch, keep_array: bool = False):
    """Image checks, then the cauliflower-leaf check. Returns (quality, leaf_check, array) for an
    image the disease model may analyse; raises Refused for anything else. With `keep_array` the
    256 x 256 model input is kept (0.8 MB), so a plain prediction needs no second decode;
    otherwise `array` is None.

    Order: file decoding -> quality gate -> cauliflower-leaf validator (blank and tiny
    images are refused before it; for dark, overexposed or blurry images its verdict
    decides which refusal the user sees). The validator runs
    in the similarity service (it needs the DINOv2 embedding); if that check cannot be
    made, the upload is refused (fail closed) rather than analysed unverified."""
    predictor = services()["predictor"]
    array = None
    with predictor.slot():  # decoding is memory-heavy: one at a time, like inference
        with sw.stage("quality_gate"):
            img = _decode(data)
            quality = assess_quality(img)
        if keep_array:
            with sw.stage("preprocess"):
                array = predictor.preprocess(img)
        del img
    refusal = suitability_outcome(quality)

    # Blank or tiny images carry nothing to check: refused straight away.
    if refusal and {p.code for p in quality.problems} & UNCHECKABLE:
        raise _refusal(refusal["status"], quality, refusal)
    # Outside the slot: waiting for a waking service must not block other users' analyses.
    with sw.stage("leaf_check"):
        leaf = services()["similarity"].verify(data, file.filename, file.mimetype)
    # A dark, overexposed or blurry image that is clearly not a cauliflower leaf (a white
    # document, a screenshot) is told so, rather than given photo-quality advice.
    if refusal and not (leaf.get("checked") and not leaf["is_cauliflower_leaf"]):
        raise _refusal(refusal["status"], quality, refusal)
    if not leaf.get("checked"):
        refuse(LEAF_CHECK_UNAVAILABLE, 503, "leaf_check_unavailable")
    if not leaf["is_cauliflower_leaf"]:
        out = not_cauliflower_outcome(leaf)
        raise _refusal(out["status"], quality, out)
    return quality, leaf, array


# ------------------------------------------------------------------ analysis
@bp.post("/verify")
@rate_limited("RATE_LIMIT_PREDICT")
def verify():
    """Only the image checks and the cauliflower-leaf check. The Analyze page calls this
    first so it can show each step; /api/predict repeats the same checks itself."""
    sw = Stopwatch()
    try:
        file, data = _read_upload()
        quality, leaf, _ = _screen(file, data, sw)
    except Refused as r:
        return r.response()
    return jsonify(success=True, status="cauliflower_leaf", leaf_check=leaf, quality=quality.to_dict(),
                   request_id=current_request_id(), timing_ms=sw.as_dict())


def _run_model(predictor, data: bytes, sw: Stopwatch, explain: bool, search_as: tuple[str, str] | None):
    """Decode again (the first decode was released to keep memory low) and classify.
    With `search_as=(filename, mimetype)` the similarity search starts alongside the CNN.
    Returns (image, prediction, similarity_job, backend)."""
    with sw.stage("decode"):
        img = _decode(data)
    job = None
    try:
        if search_as:
            job = _similarity_pool.submit(services()["similarity"].search, data, *search_as)
        prediction = predictor.predict(img, explain=explain)
    except ModelUnavailableError:
        refuse("The disease detection model is not available on this server yet.", 503, "model_unavailable")
    except Exception:  # noqa: BLE001 - never expose internals
        log.exception("Analysis failed")
        refuse("We couldn't process this image. Please try another photo.", 500, "analysis_failed")
    sw.merge(prediction.pop("timing_ms", None))
    backend = prediction.pop("backend", None)
    return img, prediction, job, backend


def _inference_info(backend: str | None) -> dict:
    return {"backend": backend, "model": services()["predictor"].model_path.name}


def _analyse_full(file, data: bytes, quality, leaf, explain: bool, k: int, sw: Stopwatch):
    predictor, kb = services()["predictor"], services()["kb"]
    with predictor.slot():
        img, prediction, job, backend = _run_model(predictor, data, sw, explain, (file.filename, file.mimetype))
        # The two additional analyses are independent of the classifier: a failure
        # in either is reported in its own section and never breaks the prediction.
        with sw.stage("severity"):
            severity_result = _severity(img)
        del img
        with sw.stage("similarity_wait"):
            similar = _similar(job, services()["similarity"].timeout)
    result = apply_second_opinion(build_result(prediction, quality, kb, k), prediction, second_opinion(prediction, similar))
    return {"success": True, "quality": quality.to_dict(), "leaf_check": leaf, "severity": severity_result,
            "similar_images": similar, "inference": _inference_info(backend), **result}


def _analyse_compact(array, quality, leaf, k: int, sw: Stopwatch):
    """Class, confidence and decision only: no Grad-CAM, affected-area or similarity work, and no
    second decode (the model input was prepared while screening)."""
    predictor = services()["predictor"]
    try:
        with predictor.slot():
            prediction = predictor.predict_array(array)
    except ModelUnavailableError:
        refuse("The disease detection model is not available on this server yet.", 503, "model_unavailable")
    except Exception:  # noqa: BLE001 - never expose internals
        log.exception("Analysis failed")
        refuse("We couldn't process this image. Please try another photo.", 500, "analysis_failed")
    sw.merge(prediction.pop("timing_ms", None))
    backend = prediction.pop("backend", None)
    return {"success": True, "status": "ok", **compact_result(prediction, k), "quality": quality.to_dict(),
            "leaf_check": leaf, "inference": _inference_info(backend), "disclaimer": DISCLAIMER}


def _cacheable(payload: dict, compact: bool) -> bool:
    if compact:
        return True
    sim = payload.get("similar_images") or {}
    # A transient similarity failure must not be remembered; "not set up" is permanent.
    return bool(sim.get("available")) or sim.get("reason") == NOT_CONFIGURED


def _ok(payload: dict, sw: Stopwatch, cached: bool):
    body = {**payload, "request_id": current_request_id(), "timing_ms": sw.as_dict(), "cached": cached}
    result = body["result"]
    annotate(decision=body["decision"], disease=result["label"], confidence_level=result["confidence_level"],
             cached=cached, backend=(body.get("inference") or {}).get("backend"))
    return jsonify(body)


@bp.post("/predict")
@rate_limited("RATE_LIMIT_PREDICT")
def predict():
    sw = Stopwatch()
    compact = request.args.get("compact", "0") == "1"
    explain = request.args.get("explain", "1") != "0" and not compact
    cache = services()["cache"]
    try:
        k = _top_k()
        with sw.stage("read_upload"):
            file, data = _read_upload()
        key = fingerprint(data, "compact" if compact else ("full" if explain else "plain"), str(k))
        if (hit := cache.get(key)) is not None:
            return _ok(hit, sw, cached=True)
        # The disease model only runs on images that pass every check, including
        # the cauliflower-leaf validator, whatever the client did before.
        quality, leaf, array = _screen(file, data, sw, keep_array=compact)
        payload = (_analyse_compact(array, quality, leaf, k, sw) if compact
                   else _analyse_full(file, data, quality, leaf, explain, k, sw))
    except Refused as r:
        return r.response()
    if _cacheable(payload, compact):
        cache.put(key, payload)
    return _ok(payload, sw, cached=False)


def _safe_name(name: str | None) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", (name or "image").rsplit("/", 1)[-1].rsplit("\\", 1)[-1])[:80]


@bp.post("/predict/batch")
@rate_limited("RATE_LIMIT_PREDICT")
def predict_batch():
    """Classify several images with ONE forward pass through the model.

    Each image gets the same checks as /api/predict (format, quality, cauliflower-leaf
    validator); images that fail are reported in their own entry and do not affect the rest.
    No Grad-CAM, affected-area or similarity work is done (compact result only)."""
    sw = Stopwatch()
    cfg = current_app.config
    predictor = services()["predictor"]
    try:
        k = _top_k()
        files = [f for f in request.files.getlist("images") if f is not None]
        if not files:
            refuse("No images were uploaded. Send one or more files in the 'images' field.", 400, "no_file")
        if len(files) > cfg["BATCH_MAX_IMAGES"]:
            refuse(f"Too many images: at most {cfg['BATCH_MAX_IMAGES']} per request.", 400, "too_many_images")
        if not consume_rate_limit("RATE_LIMIT_PREDICT", len(files) - 1):  # a batch of N costs N requests
            return jsonify(success=False, error="Too many requests. Please wait a moment and try again.",
                           code="rate_limited", decision="error", request_id=current_request_id()), 429
        try:
            predictor.load()
        except ModelUnavailableError:
            refuse("The disease detection model is not available on this server yet.", 503, "model_unavailable")

        items: list[dict | None] = [None] * len(files)
        ready: list[tuple[int, object, object]] = []  # (index, preprocessed array, quality)
        for i, f in enumerate(files):
            name = _safe_name(f.filename)
            try:
                data = _read_file(f)
                quality, _leaf, arr = _screen(f, data, sw, keep_array=True)
                ready.append((i, arr, quality))
            except Refused as r:
                p = r.payload
                items[i] = {"index": i, "filename": name, "success": False, "decision": "rejected",
                            "code": p.get("code"), "message": p.get("message") or p.get("error"),
                            "http_status": r.status}
        info = {"backend": predictor.backend, "inference_ms": 0.0, "images": 0}
        if ready:
            try:
                with predictor.slot(), sw.stage("inference"):
                    predictions, info = predictor.predict_batch([arr for _, arr, _ in ready])
            except Exception:  # noqa: BLE001 - never expose internals
                log.exception("Batch analysis failed")
                refuse("We couldn't process these images. Please try again.", 500, "analysis_failed")
            for (i, _arr, quality), prediction in zip(ready, predictions):
                items[i] = {"index": i, "filename": _safe_name(files[i].filename), "success": True, "status": "ok",
                            **compact_result(prediction, k),
                            "quality": {"acceptable": True, "warnings": [w.to_dict() for w in quality.warnings]}}
    except Refused as r:
        return r.response()

    counts = {d: sum(1 for it in items if it["decision"] == d) for d in ("accepted", "uncertain", "rejected")}
    annotate(decision="batch", cached=False, backend=info["backend"], images=len(items), **{f"n_{d}": n for d, n in counts.items()})
    return jsonify(success=True, count=len(items), **counts, results=items, disclaimer=DISCLAIMER,
                   inference={"backend": info["backend"], "model": predictor.model_path.name,
                              "batched_forward_pass_ms": info["inference_ms"]},
                   request_id=current_request_id(), timing_ms=sw.as_dict())


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


# ------------------------------------------------------------------ knowledge
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


# ------------------------------------------------------------------ model & service
@lru_cache(maxsize=1)
def _metadata() -> dict:
    return json.loads((BASE_DIR / "models" / "model_metadata.json").read_text(encoding="utf-8"))


@bp.get("/model")
def model_info():
    """What the model is, how it was evaluated, and how it is served."""
    meta = _metadata()
    p = services()["predictor"]
    evaluation = meta.get("test_evaluation", {})
    return jsonify(
        success=True,
        model={"type": meta["model_type"], "classes": meta["classes"], "input_size": meta["input_size"],
               "parameters": meta["parameters"], "preprocessing": meta["preprocessing"], "created": meta.get("created"),
               "files": {"keras": p.model_path.name, "onnx": p.onnx_path.name if p.onnx_path else None}},
        serving={"mode": p.mode, "model_loaded": p.is_loaded, "plain_prediction_backend": p.backend,
                 "gradcam_backend": p.gradcam_backend, "onnx": p.onnx_status},
        dataset={"name": meta["dataset"], "split": meta["split"], "split_method": meta["split_method"]},
        evaluation={"set": "held-out test split (not used for training or model selection)",
                    "images": evaluation.get("images"), "accuracy": meta["test_accuracy"],
                    "macro_precision": meta["test_precision"], "macro_recall": meta["test_recall"],
                    "macro_f1": meta["test_f1"], "per_class": evaluation.get("per_class"),
                    "confusion_matrix": evaluation.get("confusion_matrix"),
                    "train_accuracy": meta["train_accuracy"], "validation_accuracy": meta["val_accuracy"]},
        calibration=meta.get("calibration"),
        reliability={"uncertain_below": LOW_SCORE_THRESHOLD, "high_from": HIGH_SCORE_THRESHOLD,
                     "decisions": {"accepted": f"confidence >= {LOW_SCORE_THRESHOLD:.2f}",
                                   "uncertain": f"confidence < {LOW_SCORE_THRESHOLD:.2f}",
                                   "rejected": "refused before classification (invalid file, poor quality, "
                                               "not a cauliflower leaf)"},
                    "note": "Scores are over-confident (mean top score 96.8% vs 87.9% accuracy on the test set)."},
        request_id=current_request_id())


@bp.get("/stats")
def stats():
    """Request counts, decisions and latency percentiles since this process started."""
    svc = services()
    p = svc["predictor"]
    return jsonify(success=True, **current_app.extensions["metrics"].snapshot(), cache=svc["cache"].stats(),
                   inference={"mode": p.mode, "model_loaded": p.is_loaded, "plain_prediction_backend": p.backend,
                              "gradcam_backend": p.gradcam_backend, "onnx": p.onnx_status},
                   request_id=current_request_id())


@bp.get("/openapi.yaml")
def openapi():
    return send_from_directory(BASE_DIR / "docs", "openapi.yaml", mimetype="application/yaml")


@bp.get("/health")
def health():
    p = services()["predictor"]
    return jsonify(
        success=True,
        status="ok",
        model_loaded=p.is_loaded,
        model_file_present=p.artifact_present,
        classes=CLASS_NAMES,
        confidence_tiers={
            "uncertain": f"< {LOW_SCORE_THRESHOLD:.2f}",
            "moderate": f"{LOW_SCORE_THRESHOLD:.2f} - {HIGH_SCORE_THRESHOLD:.2f}",
            "high": f">= {HIGH_SCORE_THRESHOLD:.2f}",
        },
        tier_labels={k: v[0] for k, v in CONFIDENCE_TIERS.items()},
        inference_backend=p.backend,
        gradcam_backend=p.gradcam_backend,
        onnx=p.onnx_status,
        # Never calls the other service: this endpoint is the platform's health check.
        similarity_search=services()["similarity"].configured,
        # Analyses need the cauliflower-leaf check, which runs in the similarity service.
        leaf_validation=services()["similarity"].configured,
        similarity_service=services()["similarity"].last_status,
    )
