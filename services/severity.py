"""Estimated visible affected area: colour-based segmentation of symptom-coloured tissue.

The dataset has image-level class labels only, with no lesion masks, so no
segmentation model can be trained or validated here. This module uses the
classic colour-threshold approach from plant-disease image analysis instead:

  1. Leaf region: pixels with healthy-leaf colour (green to blue-green), closed
     morphologically and with enclosed holes filled, so lesions inside the
     leaf are counted as part of it.
  2. Symptom-coloured pixels inside the leaf region: yellow (chlorosis) and
     tan/brown (necrosis). Very dark pixels are ignored (mostly shadows).
  3. Affected share = symptom pixels / leaf-region pixels.

It is NOT the Grad-CAM map and it does not use the classifier. It measures
colour, so soil, shadows, dead leaves or a cauliflower head in the frame can
be counted, and pale symptoms such as downy growth can be missed. The severity
bands are a display convention, not validated agricultural severity classes.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

WORK_SIZE = 384  # longest side used for the estimate

# Hue in degrees, saturation/value in [0, 1].
# Cauliflower leaves are glaucous (blue-grey), so "healthy" includes low-saturation blue-green.
HEALTHY = dict(h=(62, 215), s_min=0.06, v_min=0.20)
YELLOW = dict(h=(40, 62), s_min=0.22, v_min=0.35)
BROWN = dict(h=(0, 40), s_min=0.12, v_min=0.15, v_max=0.92)
# The leaf region must cover enough of the photo for a percentage to mean anything.
MIN_LEAF_FRACTION = 0.12
# Share of bright, low-saturation cream pixels above which the photo most likely
# shows the cauliflower head (curd) rather than a leaf.
CURD_FRACTION = 0.18

BANDS = [(10, "Low"), (30, "Moderate"), (60, "High"), (100.01, "Very high")]
BAND_NOTE = "0-10% Low, 10-30% Moderate, 30-60% High, above 60% Very high (display convention, not a validated scale)"


class SeverityError(ValueError):
    """The estimate could not be produced for this image."""


@dataclass
class SeverityEstimate:
    affected_percent: float
    band: str
    leaf_fraction: float
    mask_png: str
    overlay_jpg: str

    def to_dict(self) -> dict:
        return {
            "available": True,
            "method": "Colour-based estimate of visible symptom-coloured tissue",
            "affected_percent": round(self.affected_percent, 1),
            "band": self.band,
            "band_note": BAND_NOTE,
            "leaf_fraction": round(self.leaf_fraction, 3),
            "mask": self.mask_png,
            "overlay": self.overlay_jpg,
        }


def band_for(percent: float) -> str:
    for limit, name in BANDS:
        if percent < limit:
            return name
    return BANDS[-1][1]


def _hsv(img: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    return hsv[..., 0] * 360.0, hsv[..., 1], hsv[..., 2]


def _in(h, s, v, rule) -> np.ndarray:
    lo, hi = rule["h"]
    m = (h >= lo) & (h < hi) & (s >= rule["s_min"]) & (v >= rule["v_min"])
    if "v_max" in rule:
        m &= v <= rule["v_max"]
    return m


def _to_img(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255, "L")


def _window_extreme(a: np.ndarray, size: int, use_max: bool) -> np.ndarray:
    """Maximum (or minimum) of every size x size window of a uint8 image, windows clipped at the border.

    This is what Pillow's MaxFilter / MinFilter(size) returns, but Pillow sorts size x size values
    for every pixel (about 60 ms for the 15 x 15 kernel on a 384 px photo). A square maximum or
    minimum is separable, so two passes of `size` vector operations give the same result in about
    a millisecond. Padding with 0 (for the maximum) or 255 (for the minimum) clips the windows."""
    r = size // 2
    h, w = a.shape
    pick = np.maximum if use_max else np.minimum
    padded = np.pad(a, r, mode="constant", constant_values=0 if use_max else 255)
    vertical = padded[0:h]
    for dy in range(1, size):
        vertical = pick(vertical, padded[dy:dy + h])
    window = vertical[:, 0:w]
    for dx in range(1, size):
        window = pick(window, vertical[:, dx:dx + w])
    return window


def _close(mask: np.ndarray, size: int) -> np.ndarray:
    a = mask.astype(np.uint8) * 255
    return _window_extreme(_window_extreme(a, size, True), size, False) > 127


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Pixels not reachable from the image border through non-mask pixels become mask."""
    padded = Image.new("L", (mask.shape[1] + 2, mask.shape[0] + 2), 0)
    padded.paste(_to_img(mask), (1, 1))
    ImageDraw.floodfill(padded, (0, 0), 128)  # flood the outside background
    arr = np.asarray(padded)[1:-1, 1:-1]
    return arr != 128


def _encode(img: Image.Image, fmt: str) -> str:
    buf = io.BytesIO()
    if fmt == "PNG":
        # compress_level 3 is ~10x faster than optimize=True for a 14% larger, pixel-identical file.
        img.save(buf, format="PNG", compress_level=3)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    img.convert("RGB").save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def segment(img: Image.Image) -> tuple[np.ndarray, np.ndarray, Image.Image]:
    """Return (leaf_region, affected) boolean masks at working size, plus the resized RGB image."""
    leaf, affected, small, _ = _segment(img)
    return leaf, affected, small


def _segment(img: Image.Image):
    small = img.convert("RGB")
    small.thumbnail((WORK_SIZE, WORK_SIZE))
    h, s, v = _hsv(small)
    healthy = _in(h, s, v, HEALTHY)
    k = max(7, (max(small.size) // 25) | 1)  # odd kernel, ~4% of the image
    leaf = _fill_holes(_close(healthy, k))
    # Very dark pixels are not counted: in field photos they are mostly shadows.
    symptom = _in(h, s, v, YELLOW) | _in(h, s, v, BROWN)
    affected = symptom & leaf & ~healthy
    return leaf, affected, small, (h, s, v)


def estimate(img: Image.Image) -> SeverityEstimate:
    if img is None or min(img.size) < 32:
        raise SeverityError("image too small")
    leaf, affected, small, (h, s, v) = _segment(img)
    leaf_fraction = float(leaf.mean())
    if leaf_fraction < MIN_LEAF_FRACTION:
        raise SeverityError("not enough leaf area is visible to estimate an affected share")
    cream = (s < 0.30) & (v > 0.60) & (h >= 25) & (h <= 75)
    if float(cream.mean()) > CURD_FRACTION:
        raise SeverityError("the photo appears to show the cauliflower head rather than a leaf")

    percent = 100.0 * float(affected.sum()) / float(leaf.sum())
    region, overlay = render_views(small, leaf, affected)
    return SeverityEstimate(percent, band_for(percent), leaf_fraction, _encode(region, "PNG"), _encode(overlay, "JPEG"))


AFFECTED_RGB = (232, 36, 36)   # estimated affected pixels
OUTLINE_RGB = (255, 214, 0)    # 1 px ring around affected spots so small spots stay visible


def render_views(small: Image.Image, leaf: np.ndarray, affected: np.ndarray) -> tuple[Image.Image, Image.Image]:
    """Two visualisations of the SAME estimate (drawing only; the percentage is computed above).

    region:  the photo in dimmed greyscale (background darker than the leaf) with the
             estimated affected pixels in solid red, so "where" is readable at a glance.
    overlay: the original colour photo with the affected pixels tinted red.
    Both draw a thin yellow ring just outside each affected spot; the ring is for
    visibility only and is not part of the estimated area.
    """
    rgb = np.asarray(small.convert("RGB"), dtype=np.float32)
    ring = _ring(affected)

    grey = np.asarray(small.convert("L"), dtype=np.float32)[..., None].repeat(3, axis=2)
    region = np.where(leaf[..., None], grey * 0.55 + 40, grey * 0.25)
    region[affected] = AFFECTED_RGB
    region[ring] = OUTLINE_RGB

    overlay = rgb.copy()
    overlay[affected] = overlay[affected] * 0.35 + np.array(AFFECTED_RGB, dtype=np.float32) * 0.65
    overlay[ring] = OUTLINE_RGB

    to_img = lambda a: Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")  # noqa: E731
    return to_img(region), to_img(overlay)


def _ring(mask: np.ndarray) -> np.ndarray:
    """Pixels just outside the mask (a thin outline), empty when nothing is affected."""
    if not mask.any():
        return np.zeros_like(mask)
    grown = np.asarray(_to_img(mask).filter(ImageFilter.MaxFilter(3))) > 127
    return grown & ~mask
