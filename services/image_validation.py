"""Upload validation, image-quality checks and the leaf-suitability gate.

Everything here runs before the classifier. The goal is to refuse images the
model cannot say anything meaningful about, instead of forcing them into one
of its four leaf classes.

Thresholds were set from the 656 images of the training dataset (see
training/README.md): every hard rejection limit sits outside the range observed
on real dataset photos, so no dataset image is rejected. tests/test_real_model.py
checks this whenever the dataset is available.

  metric (on a 512 px greyscale copy)   dataset min..max   reject      warn
  mean brightness (0-255)               37 .. 168          < 25        < 45
  blown-out pixels (>= 250)             0 .. 1.9 %         > 40 %      > 15 %
  grey-level std (blank frame)          >= 20.8            < 4         -
  Laplacian variance (sharpness)        22.7 .. 3686       < 10        < 40
  shortest side (px)                    250 .. 4032        < 128       < 224

The leaf gate is a heuristic, not a trained leaf detector: it checks that a
reasonable share of pixels are plant-coloured and that the texture is not
noise-like. Thresholds come from the training split (min plant-coloured
fraction 0.094; max texture variance 4563 at 256 px). It reliably rejects blank
frames, noise, screenshots and most non-plant scenes, but it cannot tell a
cauliflower leaf from another green leaf.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, UnidentifiedImageError

DARK_REJECT, DARK_WARN = 25.0, 45.0
CLIPPED_REJECT, CLIPPED_WARN = 0.40, 0.15
BLUR_REJECT, BLUR_WARN = 10.0, 40.0
MIN_SIDE_REJECT, MIN_SIDE_WARN = 128, 224
BLANK_STD = 4.0  # grey-level standard deviation of an effectively uniform frame

MIN_VEGETATION_FRACTION = 0.08  # below the dataset minimum (0.094)
MAX_TEXTURE_VARIANCE = 6000      # above the dataset maximum (4563)

ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/pjpeg", "image/png", "image/webp"}
# Clients that cannot tell the type send these; the real format is still checked from the bytes.
GENERIC_MIME_TYPES = {"", "application/octet-stream"}


class ImageValidationError(ValueError):
    """An upload that must be rejected. The message is safe to show to users."""

    def __init__(self, message: str, code: str = "invalid_image", status: int = 422):
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass
class Issue:
    code: str
    message: str

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message}


@dataclass
class QualityReport:
    width: int
    height: int
    brightness: float
    clipped_fraction: float
    sharpness: float
    vegetation_fraction: float
    texture: float
    problems: list[Issue] = field(default_factory=list)  # reasons to refuse analysis
    warnings: list[Issue] = field(default_factory=list)  # analysed, but flagged

    @property
    def acceptable(self) -> bool:
        return not self.problems

    @property
    def looks_like_leaf(self) -> bool:
        return self.vegetation_fraction >= MIN_VEGETATION_FRACTION and self.texture <= MAX_TEXTURE_VARIANCE

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "brightness": round(self.brightness, 1),
            "clipped_fraction": round(self.clipped_fraction, 4),
            "sharpness": round(self.sharpness, 1),
            "vegetation_fraction": round(self.vegetation_fraction, 3),
            "acceptable": self.acceptable,
            "looks_like_leaf": self.looks_like_leaf,
            "problems": [p.to_dict() for p in self.problems],
            "warnings": [w.to_dict() for w in self.warnings],
        }


def allowed_extension(filename: str, allowed: set[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def allowed_mimetype(mimetype: str | None) -> bool:
    m = (mimetype or "").split(";")[0].strip().lower()
    return m in ALLOWED_MIME_TYPES or m in GENERIC_MIME_TYPES


def load_image(data: bytes, allowed_formats: set[str], max_pixels: int) -> Image.Image:
    """Decode bytes into an RGB PIL image, verifying the real format from the content."""
    if not data:
        raise ImageValidationError("The uploaded file is empty.", "empty_file", 400)
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        probe = Image.open(io.BytesIO(data))
        fmt = probe.format
        probe.verify()  # detects truncated / corrupted files
        img = Image.open(io.BytesIO(data))
        img.load()
    except Image.DecompressionBombError as exc:
        raise ImageValidationError("The image resolution is too large to process.", "too_many_pixels") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError(
            "We couldn't process this image. Please try another photo.", "corrupt_image"
        ) from exc

    if fmt not in allowed_formats:
        raise ImageValidationError("Please upload a JPG, PNG, or WEBP image.", "unsupported_format", 415)

    # Do NOT rotate pixels by EXIF here: the model was trained on images decoded by
    # TensorFlow, which ignores EXIF orientation. The tag is kept so explanation
    # images can be rotated to match the upright photo the browser displays.
    orientation = img.getexif().get(0x0112, 1)
    if img.mode != "RGB":
        if img.mode in ("RGBA", "LA", "P"):  # flatten transparency onto white, not black
            img = img.convert("RGBA")
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
    img.info["exif_orientation"] = orientation
    return img


def _laplacian_variance(gray: np.ndarray) -> float:
    lap = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(lap.var())


def assess_quality(img: Image.Image) -> QualityReport:
    gray = img.convert("L")
    gray.thumbnail((512, 512))
    g = np.asarray(gray, dtype=np.float32)
    brightness = float(g.mean())
    clipped = float((g >= 250).mean())
    sharpness = _laplacian_variance(g)

    small = img.convert("RGB")
    small.thumbnail((256, 256))
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32) / 255.0
    hue, sat, val = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    vegetation = float(((hue >= 40) & (hue <= 200) & (sat >= 0.12) & (val >= 0.12)).mean())
    texture = _laplacian_variance(np.asarray(small.convert("L"), dtype=np.float32))

    report = QualityReport(img.width, img.height, brightness, clipped, sharpness, vegetation, texture)
    if brightness < DARK_REJECT:
        report.problems.append(Issue("too_dark", "Image is too dark. Take the photo in daylight or better lighting."))
    elif brightness < DARK_WARN:
        report.warnings.append(Issue("too_dark", "The image is fairly dark; better lighting usually gives a more reliable result."))

    if clipped > CLIPPED_REJECT:
        report.problems.append(Issue("overexposed", "Image is overexposed. Avoid direct sunlight or glare on the leaf."))
    elif clipped > CLIPPED_WARN:
        report.warnings.append(Issue("overexposed", "Parts of the image are overexposed; avoid glare on the leaf."))

    if float(g.std()) < BLANK_STD:
        # A flat frame: report exposure if that explains it, otherwise call it blank.
        if not report.problems:
            report.problems.append(Issue("blank_image", "The image appears to be blank or a single flat colour."))
        return report

    side = min(img.size)
    if side < MIN_SIDE_REJECT:
        report.problems.append(Issue("low_resolution", f"Image resolution is too low ({img.width}x{img.height} px). "
                                     f"Please use a photo at least {MIN_SIDE_REJECT} px on each side."))
    elif side < MIN_SIDE_WARN:
        report.warnings.append(Issue("low_resolution", "The image is quite small; a larger photo gives the model more detail."))

    if sharpness < BLUR_REJECT:
        report.problems.append(Issue("too_blurry", "Image is too blurry. Hold the camera steady and focus on the leaf."))
    elif sharpness < BLUR_WARN:
        report.warnings.append(Issue("too_blurry", "The image may be slightly blurry; a sharper photo is more reliable."))
    return report
