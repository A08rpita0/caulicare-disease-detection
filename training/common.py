"""Shared constants and helpers for the CauliCare training pipeline."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "data" / "raw" / "Original Dataset"
SPLIT_FILE = ROOT / "data" / "splits.json"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

# Alphabetical folder order == image_dataset_from_directory order in the notebook.
CLASS_NAMES = ["Bacterial spot rot", "Black Rot", "Downy Mildew", "No disease"]
IMAGE_SIZE = 256
SEED = 42
EXTS = {".jpg", ".jpeg", ".png"}


def list_images(root: Path = DATASET_DIR) -> list[tuple[Path, int]]:
    found = sorted(d.name for d in root.iterdir() if d.is_dir())
    assert found == CLASS_NAMES, f"Class folders {found} != {CLASS_NAMES}"
    items = []
    for idx, name in enumerate(CLASS_NAMES):
        for p in sorted((root / name).iterdir()):
            if p.suffix.lower() in EXTS:
                items.append((p, idx))
    return items


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def dhash(img: Image.Image, size: int = 8) -> int:
    """Difference hash: robust to resizing/re-encoding, used for near-duplicate detection."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def load_rgb_256(path: Path) -> np.ndarray:
    """Exactly the app's preprocessing: RGB, bilinear 256x256, float32 0-255 (model rescales)."""
    img = Image.open(path)
    img.load()
    img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)
