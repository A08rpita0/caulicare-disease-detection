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
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

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


class Predictor:
    def __init__(self, model_path: Path, model_url: str = "", max_concurrent: int = 1):
        self.model_path = Path(model_path)
        self.model_url = model_url
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
        return self._model is not None

    def _download(self) -> None:
        log.info("Downloading model from MODEL_URL")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.model_path.with_suffix(".part")
        with urllib.request.urlopen(self.model_url, timeout=120) as resp, open(tmp, "wb") as fh:  # noqa: S310
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        tmp.replace(self.model_path)

    def load(self):
        """Load the model once (thread-safe) and cache it for all requests."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
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
                self.load_error = None
                log.info("Model loaded from %s", self.model_path)
            except Exception as exc:  # noqa: BLE001 - surfaced as a clean 503
                self.load_error = str(exc)
                log.error("Model failed to load: %s", exc)
                raise ModelUnavailableError(self.load_error) from exc
        return self._model

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

    # ------------------------------------------------------------- inference
    @staticmethod
    def preprocess(img: Image.Image) -> np.ndarray:
        return preprocess_image(img)

    def predict(self, img: Image.Image, explain: bool = True) -> dict:
        model = self.load()
        batch = self.preprocess(img)
        probs, heatmap = self._forward(model, batch, explain)

        result = summarize(probs)
        if heatmap is not None:
            orient = img.info.get("exif_orientation", 1)
            result["explanation"] = {
                "method": "Grad-CAM",
                "layer": model.layers[self._last_conv_index].name,
                "heatmap": _to_data_url(_orient(_colorize(heatmap, (IMAGE_SIZE, IMAGE_SIZE)), orient)),
                "overlay": _to_data_url(_orient(_overlay(img, heatmap), orient)),
            }
        return result

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
        "confidence_level": tier,
        "confidence_label": label,
        "confidence_message": message,
        "runner_up": {"disease": CLASS_NAMES[second], "probability": round(runner, 4)},
        "margin": round(score - runner, 4),
        "confusable_close_call": pair and runner >= CLOSE_CALL_MIN_RUNNER_UP,
        "all_predictions": {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in order},
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
