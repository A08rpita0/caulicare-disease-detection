"""TensorFlow inference service for the CauliCare CNN.

Preprocessing contract (verified from notebooks/Model.ipynb):
  * Training images came from `image_dataset_from_directory(image_size=(256, 256))`,
    i.e. RGB, resized with tf.image.resize(bilinear, antialias=False) to 256x256
    without preserving aspect ratio, float pixel values in [0, 255]. (Verified by
    reproducing the notebook's printed pixel values from the dataset file
    "Black Rot(33).jpg" to within 3e-8.) PIL's BILINEAR antialiases when
    downscaling and gives different pixels, so `resize_bilinear_like_tf` below
    reimplements TF's kernel in NumPy. It is bit-identical to tf.image.resize on
    all 656 dataset images, but reads only the source pixels it needs instead of
    converting the whole photo to float32 (~144 MB for a 12 MP photo), which is
    what lets the app fit in a 512 MB free hosting instance.
  * The model itself starts with `Resizing(256, 256)` + `Rescaling(1/255)` layers,
    followed by RandomFlip/RandomRotation (inactive at inference).
  So the correct input is an RGB float32 tensor of raw 0-255 values at 256x256.
  Dividing by 255 before calling the model would normalise twice and silently
  corrupt predictions.
"""
from __future__ import annotations

import base64
import io
import logging
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from services.gradcam_onnx import cam_from_features, load_sidecar, sidecar_path
from services.onnx_export import sha256_file

log = logging.getLogger(__name__)

# The ONNX copy must reproduce the Keras probabilities to this tolerance on a fixed probe
# batch at start-up; otherwise (for example after retraining only the .h5) it is not used.
ONNX_PARITY_TOLERANCE = 1e-3

# Order is the alphabetical folder order used by image_dataset_from_directory,
# confirmed by the notebook output: dataset.class_names.
CLASS_NAMES = ["Bacterial spot rot", "Black Rot", "Downy Mildew", "No disease"]
IMAGE_SIZE = 256
# Chosen on the VALIDATION split of the trained model1.h5 (reports/, training/README):
# below 0.80 only 42% of validation predictions were correct; >=0.95 is "high".
LOW_SCORE_THRESHOLD = 0.80
HIGH_SCORE_THRESHOLD = 0.95
# Most held-out test errors were confusions between these two classes.
CONFUSABLE = {"Bacterial spot rot", "Downy Mildew"}
# When the runner-up of a confusable pair gets at least this probability, the
# result page shows a side-by-side comparison of the two conditions.
CLOSE_CALL_MIN_RUNNER_UP = 0.10

CONFIDENCE_TIERS = {
    "uncertain": ("Uncertain", "Low confidence - this result may be unreliable."),
    "moderate": ("Moderate confidence", "Moderate confidence - consider retaking the photo or comparing symptoms."),
    "high": ("High confidence", "High confidence - based on the uploaded image."),
}


def confidence_tier(score: float) -> str:
    """<0.80 uncertain, 0.80-0.95 moderate, >=0.95 high."""
    if score < LOW_SCORE_THRESHOLD:
        return "uncertain"
    return "moderate" if score < HIGH_SCORE_THRESHOLD else "high"


class ModelUnavailableError(RuntimeError):
    """The trained model artifact is missing or failed to load."""


BACKENDS = ("auto", "onnx", "tensorflow")


class Predictor:
    """The CNN, served by Keras, by ONNX Runtime, or both.

    backend="tensorflow"  Keras only (Grad-CAM through TensorFlow's autodiff).
    backend="auto"        Keras, plus the ONNX model when it reproduces the Keras model. Verified
                          ONNX is used for every request: plain predictions, batches and Grad-CAM.
    backend="onnx"        ONNX Runtime only. TensorFlow is never imported: a few hundred MB less
                          memory and sub-second start-up. The ONNX file must record the SHA-256 of
                          the model file it was exported from; a mismatch makes the model unavailable
                          instead of silently serving a stale network.
    """

    def __init__(self, model_path: Path, model_url: str = "", max_concurrent: int = 1,
                 onnx_path: Path | str | None = None, onnx_threads: int = 1, backend: str = "auto"):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, not {backend!r}")
        self.mode = backend
        self.model_path = Path(model_path)
        self.model_url = model_url
        self.onnx_path = Path(onnx_path) if onnx_path and backend != "tensorflow" else None
        self.onnx_threads = max(1, int(onnx_threads))
        self.onnx_status = "disabled" if self.onnx_path is None else "not loaded"
        self._onnx = None
        self._gradcam = None
        self._model = None
        self._last_conv_index = None
        self._lock = threading.Lock()
        # Decoding a phone photo, inference and Grad-CAM together add ~150+ MB on top of
        # the idle TensorFlow footprint. On a 512 MB instance two at once exceed the
        # limit, so the API holds a slot for the whole image pipeline (see slot()).
        self._infer_slots = threading.BoundedSemaphore(max(1, int(max_concurrent)))
        self.load_error: str | None = None

    def slot(self):
        """Context manager limiting how many images are processed at the same time."""
        return self._infer_slots

    # ------------------------------------------------------------------ loading
    @property
    def is_loaded(self) -> bool:
        return self._model is not None or (self.mode == "onnx" and self._onnx is not None)

    @property
    def artifact_present(self) -> bool:
        path = self.onnx_path if self.mode == "onnx" else self.model_path
        return path is not None and Path(path).exists()

    @property
    def backend(self) -> str:
        """Backend used for plain (no Grad-CAM) predictions."""
        return "onnxruntime" if self._onnx is not None else "tensorflow"

    @property
    def gradcam_backend(self) -> str:
        if self._onnx is not None and self._gradcam is not None:
            return "onnxruntime"
        return "tensorflow" if self._model is not None else "none"

    def _download(self) -> None:
        log.info("Downloading model from MODEL_URL")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.model_path.with_suffix(".part")
        with urllib.request.urlopen(self.model_url, timeout=120) as resp, open(tmp, "wb") as fh:  # noqa: S310
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        tmp.replace(self.model_path)

    def _runtime(self):
        return self._model if self._model is not None else self._onnx

    def load(self):
        """Load the model once (thread-safe) and cache it for all requests."""
        if self.is_loaded:
            return self._runtime()
        with self._lock:
            if self.is_loaded:
                return self._runtime()
            try:
                if self.mode == "onnx":
                    self._load_onnx_only()
                else:
                    self._load_keras()
                self.load_error = None
            except Exception as exc:  # noqa: BLE001 - surfaced as a clean 503
                self.load_error = str(exc)
                log.error("Model failed to load: %s", exc)
                raise ModelUnavailableError(self.load_error) from exc
            if self.mode == "auto":
                self.load_onnx()
        return self._runtime()

    def _load_keras(self) -> None:
        if not self.model_path.exists() and self.model_url:
            self._download()
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at '{self.model_path}'. "
                "Export it from notebooks/Model.ipynb and place it there, or set MODEL_URL."
            )
        import tensorflow as tf  # imported lazily: keeps app start fast & tests light

        model = tf.keras.models.load_model(self.model_path, compile=False)
        self._validate(model)
        self._model = model
        log.info("Model loaded from %s", self.model_path)

    def _session(self):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.onnx_threads
        opts.inter_op_num_threads = 1
        return ort.InferenceSession(str(self.onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])

    def _read_gradcam(self, session) -> dict | None:
        """Grad-CAM parameters next to the ONNX model, if they belong to it."""
        side = sidecar_path(self.onnx_path)
        if not side.exists():
            return None
        try:
            params = load_sidecar(side)
        except Exception as exc:  # noqa: BLE001 - Grad-CAM is optional
            log.warning("Grad-CAM parameters unreadable: %s", exc)
            return None
        recorded = session.get_modelmeta().custom_metadata_map.get("source_sha256", "")
        outputs = {o.name for o in session.get_outputs()}
        if params["source_sha256"] != recorded or not {"conv_features", "hidden"} <= outputs:
            log.warning("Grad-CAM parameters do not belong to %s; ignored", self.onnx_path.name)
            return None
        return params

    def _load_onnx_only(self) -> None:
        if self.onnx_path is None or not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at '{self.onnx_path}'. Run scripts/export_onnx.py.")
        session = self._session()
        note = "source model file absent, not verified"
        if self.model_path.exists():
            recorded = session.get_modelmeta().custom_metadata_map.get("source_sha256", "")
            if not recorded or recorded != sha256_file(self.model_path):
                raise ValueError(f"{self.onnx_path.name} was not exported from {self.model_path.name}; "
                                 "run scripts/export_onnx.py")
            note = "matches the model file"
        out = session.get_outputs()[0].shape
        if out[-1] != len(CLASS_NAMES):
            raise ValueError(f"Model outputs {out[-1]} classes, expected {len(CLASS_NAMES)}.")
        self._gradcam = self._read_gradcam(session)
        self._onnx = session
        self.onnx_status = (f"active, TensorFlow not used ({note}; Grad-CAM "
                            f"{'from ONNX activations' if self._gradcam else 'unavailable'})")
        log.info("ONNX Runtime model loaded: %s", self.onnx_path)

    def load_onnx(self) -> bool:
        """Use the ONNX model next to the Keras one if it reproduces it.

        Never raises: any problem leaves the Keras model in charge and is reported in
        `onnx_status`. Requires the Keras model to be loaded (it is the reference)."""
        if self.onnx_path is None or self._model is None:
            return False
        try:
            if not self.onnx_path.exists():
                self.onnx_status = "file not found; using TensorFlow"
                return False
            session = self._session()
            probe = np.random.default_rng(0).uniform(0, 255, (2, IMAGE_SIZE, IMAGE_SIZE, 3)).astype(np.float32)
            reference = self._model(probe, training=False).numpy()
            candidate = session.run(["probabilities"], {"image": probe})[0]
            diff = float(np.abs(reference - candidate).max())
            if diff > ONNX_PARITY_TOLERANCE:
                self.onnx_status = f"rejected: differs from the Keras model by {diff:.1e}; using TensorFlow"
                log.warning("ONNX model ignored: %s", self.onnx_status)
                return False
            gradcam, cam_note = self._read_gradcam(session), "Grad-CAM on Keras"
            if gradcam is not None:
                _, expected = self._forward(self._model, probe[:1], True)
                got = self._cam(session, gradcam, probe[:1])[1]
                if expected is not None and float(np.abs(expected - got).max()) <= ONNX_PARITY_TOLERANCE:
                    cam_note = "Grad-CAM on ONNX"
                else:
                    gradcam = None
            self._onnx, self._gradcam = session, gradcam
            self.onnx_status = f"active (max difference from Keras {diff:.1e}; {cam_note})"
            log.info("ONNX Runtime enabled: %s", self.onnx_path)
            return True
        except Exception as exc:  # noqa: BLE001 - the fast path is optional
            self.onnx_status = f"unavailable ({exc.__class__.__name__}); using TensorFlow"
            log.warning("ONNX unavailable: %s", exc)
            return False

    def _validate(self, model) -> None:
        out = model.output_shape
        if out[-1] != len(CLASS_NAMES):
            raise ValueError(f"Model outputs {out[-1]} classes, expected {len(CLASS_NAMES)}.")
        head = model.layers[-1]
        if head.__class__.__name__ != "Dense" or getattr(head.activation, "__name__", "") != "softmax":
            raise ValueError("Expected the model to end in a Dense softmax layer.")
        conv_idx = [i for i, layer in enumerate(model.layers) if layer.__class__.__name__ == "Conv2D"]
        self._last_conv_index = conv_idx[-1] if conv_idx else None

    def set_model(self, model) -> None:
        """Inject an in-memory model (used by the test-suite only)."""
        self._validate(model)
        self._model = model
        self.load_onnx()

    # ------------------------------------------------------------- inference
    @staticmethod
    def preprocess(img: Image.Image) -> np.ndarray:
        return preprocess_image(img)

    def predict(self, img: Image.Image, explain: bool = True) -> dict:
        """Class, probabilities and confidence for one image.

        Besides the prediction, the result carries `backend` ("onnxruntime" or "tensorflow")
        and `timing_ms` ({"preprocess", "inference", "explanation_render"}); the API moves both
        out of the prediction into its own response fields."""
        self.load()
        t0 = time.perf_counter()
        batch = self.preprocess(img)
        t1 = time.perf_counter()
        heatmap = layer = None
        if self._onnx is not None and (not explain or self._gradcam is not None or self._model is None):
            backend = "onnxruntime"
            if explain and self._gradcam is not None:
                try:
                    probs, heatmap = self._cam(self._onnx, self._gradcam, batch)
                    layer = self._gradcam["layer"]
                except Exception as exc:  # noqa: BLE001 - explanation is optional
                    log.warning("Grad-CAM failed, returning prediction only: %s", exc)
                    probs, heatmap = self._onnx_forward(batch)[0], None
            else:
                probs = self._onnx_forward(batch)[0]
        else:
            backend = "tensorflow"
            probs, heatmap = self._forward(self._model, batch, explain)
            if heatmap is not None:
                layer = self._model.layers[self._last_conv_index].name
        t2 = time.perf_counter()

        result = summarize(probs)
        if heatmap is not None:
            orient = img.info.get("exif_orientation", 1)
            result["explanation"] = {
                "method": "Grad-CAM",
                "layer": layer,
                "heatmap": _to_data_url(_orient(_colorize(heatmap, (IMAGE_SIZE, IMAGE_SIZE)), orient)),
                "overlay": _to_data_url(_orient(_overlay(img, heatmap), orient)),
            }
        t3 = time.perf_counter()
        result["backend"] = backend
        result["timing_ms"] = {"preprocess": round((t1 - t0) * 1000, 1), "inference": round((t2 - t1) * 1000, 1),
                               "explanation_render": round((t3 - t2) * 1000, 1)}
        return result

    def predict_array(self, batch: np.ndarray) -> dict:
        """Plain prediction (no Grad-CAM) from an already preprocessed (1, 256, 256, 3) array.
        Same result and extra fields as `predict(img, explain=False)`, minus preprocessing."""
        self.load()
        t0 = time.perf_counter()
        if self._onnx is not None:
            probs, backend = self._onnx_forward(batch)[0], "onnxruntime"
        else:
            probs, backend = self._forward(self._model, batch, False)[0], "tensorflow"
        result = summarize(probs)
        result["backend"] = backend
        result["timing_ms"] = {"inference": round((time.perf_counter() - t0) * 1000, 1)}
        return result

    def predict_batch(self, batches: list[np.ndarray]) -> tuple[list[dict], dict]:
        """Classify several preprocessed images (each shape (1, 256, 256, 3)) in ONE forward pass.

        Returns (predictions, info) where info has the backend and the pass's timing."""
        self.load()
        x = np.concatenate(batches, axis=0)
        t0 = time.perf_counter()
        if self._onnx is not None:
            probs, backend = self._onnx_forward(x), "onnxruntime"
        else:
            import tensorflow as tf

            probs, backend = self._model(tf.convert_to_tensor(x), training=False).numpy(), "tensorflow"
        ms = round((time.perf_counter() - t0) * 1000, 1)
        return [summarize(p) for p in probs], {"backend": backend, "inference_ms": ms, "images": len(batches)}

    def _onnx_forward(self, batch: np.ndarray) -> np.ndarray:
        return self._onnx.run(["probabilities"], {"image": np.ascontiguousarray(batch, dtype=np.float32)})[0]

    @staticmethod
    def _cam(session, params: dict, batch: np.ndarray):
        """(probabilities, Grad-CAM heat-map) for a one-image batch, computed from ONNX activations."""
        probs, conv, hidden = session.run(["probabilities", "conv_features", "hidden"],
                                          {"image": np.ascontiguousarray(batch, dtype=np.float32)})
        cam = cam_from_features(conv[0], hidden[0], int(np.argmax(probs[0])), params["w_hidden"],
                                params["w_out"], params["pool"])
        return probs[0], cam

    def _forward(self, model, batch: np.ndarray, explain: bool):
        import tensorflow as tf

        x = tf.convert_to_tensor(batch)
        if not explain or self._last_conv_index is None:
            return model(x, training=False).numpy()[0], None

        # Grad-CAM: run the Sequential stack layer-by-layer so we can capture the
        # last Conv2D activation and the gradient of the top class w.r.t. it.
        # As in the original method, the gradient is taken of the class score
        # *before* softmax: on confident predictions the softmax saturates and its
        # gradients vanish (~1e-7 here), which flattens the heatmap.
        try:
            head = model.layers[-1]
            with tf.GradientTape() as tape:
                h = x
                conv_out = None
                for i, layer in enumerate(model.layers[:-1]):
                    h = layer(h, training=False)
                    if i == self._last_conv_index:
                        conv_out = h
                        tape.watch(conv_out)
                logits = tf.matmul(h, head.kernel) + head.bias
                probs = tf.nn.softmax(logits)
                top = tf.argmax(probs[0])
                target = logits[:, top]
            grads = tape.gradient(target, conv_out)
            weights = tf.reduce_mean(grads, axis=(0, 1, 2))
            cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights, axis=-1)).numpy()
            cam = cam / cam.max() if cam.max() > 0 else cam
            return probs.numpy()[0], cam
        except Exception as exc:  # noqa: BLE001 - explanation is optional
            log.warning("Grad-CAM failed, returning prediction only: %s", exc)
            return model(x, training=False).numpy()[0], None


def summarize(probs) -> dict:
    """Turn a softmax vector into the prediction payload shown to users."""
    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(probs)[::-1]
    top, second = int(order[0]), int(order[1])
    score, runner = float(probs[top]), float(probs[second])
    tier = confidence_tier(score)
    label, message = CONFIDENCE_TIERS[tier]
    pair = CLASS_NAMES[top] in CONFUSABLE and CLASS_NAMES[second] in CONFUSABLE
    return {
        "disease": CLASS_NAMES[top],
        "confidence": round(score, 4),
        "confidence_percent": round(score * 100, 1),
        "confidence_level": tier,
        "confidence_label": label,
        "confidence_message": message,
        "runner_up": {"disease": CLASS_NAMES[second], "probability": round(runner, 4)},
        "margin": round(score - runner, 4),
        "confusable_close_call": pair and runner >= CLOSE_CALL_MIN_RUNNER_UP,
        "all_predictions": {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in order},
        "all_predictions_percent": {CLASS_NAMES[i]: round(float(probs[i]) * 100, 1) for i in order},
    }


def resize_bilinear_like_tf(rgb: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """uint8 HxWx3 -> float32 size x size x 3, matching TensorFlow's
    ResizeBilinear(half_pixel_centers=True) kernel, i.e.
    tf.image.resize(method="bilinear", antialias=False), in float32 arithmetic.
    Only the two source rows/columns each output pixel needs are read."""
    h, w = rgb.shape[:2]

    def axis(n_in):
        scale = np.float32(n_in) / np.float32(size)
        src = (np.arange(size, dtype=np.float32) + np.float32(0.5)) * scale - np.float32(0.5)
        fl = np.floor(src)
        lerp = (src - fl).astype(np.float32)
        lo = np.clip(fl, 0, n_in - 1).astype(np.int64)
        hi = np.clip(np.ceil(src), 0, n_in - 1).astype(np.int64)
        return lo, hi, lerp

    y0, y1, fy = axis(h)
    x0, x1, fx = axis(w)
    rows0, rows1 = rgb[y0], rgb[y1]
    tl = rows0[:, x0].astype(np.float32)
    tr = rows0[:, x1].astype(np.float32)
    bl = rows1[:, x0].astype(np.float32)
    br = rows1[:, x1].astype(np.float32)
    fx3 = fx[None, :, None]
    top = tl + (tr - tl) * fx3
    bottom = bl + (br - bl) * fx3
    return top + (bottom - top) * fy[:, None, None]


def preprocess_image(img: Image.Image) -> np.ndarray:
    """RGB -> bilinear 256x256 (identical to training) -> float32 0-255 -> batch of 1.

    No /255 here: the model's first block (Rescaling(1/255)) normalises.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return resize_bilinear_like_tf(rgb)[np.newaxis, ...]


def predict_image(model, img: Image.Image) -> dict:
    """Standalone helper (used by training/evaluate.py): class + probabilities."""
    probs = model(preprocess_image(img), training=False).numpy()[0]
    top = int(np.argmax(probs))
    return {"disease": CLASS_NAMES[top], "confidence": float(probs[top]),
            "all_predictions": {c: float(p) for c, p in zip(CLASS_NAMES, probs)}}


# ------------------------------------------------------------- heatmap utils
def _jet(v: np.ndarray) -> np.ndarray:
    """Minimal jet-style colormap for values in [0, 1] -> uint8 RGB."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _colorize(cam: np.ndarray, size) -> Image.Image:
    small = Image.fromarray((cam * 255).astype(np.uint8))
    big = np.asarray(small.resize(size, Image.BICUBIC), dtype=np.float32) / 255.0
    return Image.fromarray(_jet(big))


def _overlay(img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = img.convert("RGB").resize((IMAGE_SIZE * 2, IMAGE_SIZE * 2), Image.BILINEAR)
    return Image.blend(base, _colorize(cam, base.size), alpha)


_EXIF_OPS = {
    2: [Image.Transpose.FLIP_LEFT_RIGHT], 3: [Image.Transpose.ROTATE_180], 4: [Image.Transpose.FLIP_TOP_BOTTOM],
    5: [Image.Transpose.TRANSPOSE], 6: [Image.Transpose.ROTATE_270], 7: [Image.Transpose.TRANSVERSE],
    8: [Image.Transpose.ROTATE_90],
}


def _orient(img: Image.Image, orientation: int) -> Image.Image:
    """Apply an EXIF orientation to a rendered explanation image (display only)."""
    for op in _EXIF_OPS.get(orientation, []):
        img = img.transpose(op)
    return img


def _to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
