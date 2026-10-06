"""Visual similarity search: DINOv2 image embeddings + nearest-neighbour search.

This is a second, independent signal next to the disease classifier:
  * the CNN answers "which of the four classes does the model predict?";
  * this module answers "which indexed reference photos look most similar?".

Embeddings come from DINOv2-small (self-supervised ViT-S/14, Meta AI, Apache-2.0),
run through ONNX Runtime so no PyTorch is needed. It was not trained on
cauliflower diseases: similarity reflects general visual closeness (colour,
texture, layout), not disease probability.

Vectors are L2-normalised, so inner product = cosine similarity. Search uses a
FAISS IndexFlatIP (exact) when the faiss package is available, otherwise the
identical computation in NumPy; for a few hundred vectors both are exact and
return the same neighbours. The index is built offline by
scripts/build_image_index.py and loaded once when the similarity service starts.
Each result carries its thumbnail inline as a data URI, looked up by indexed id.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

from .validator import LeafValidator, ValidatorUnavailableError

log = logging.getLogger(__name__)

EMBEDDING_DIM = 384
_ID = re.compile(r"^[a-z0-9-]{1,64}$")


class SimilarityUnavailableError(RuntimeError):
    """The embedding model or the reference index is not installed."""


class ImageEncoder:
    """DINOv2-small ONNX encoder. Preprocessing follows the model's own
    preprocessor_config.json: shortest edge -> 256 (bicubic), centre crop 224,
    scale to [0, 1], ImageNet mean/std normalisation."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self._session = None
        self._lock = threading.Lock()
        cfg_path = self.model_path.with_name("preprocessor_config.json")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        self.short_edge = int(cfg.get("size", {}).get("shortest_edge", 256))
        self.crop = int(cfg.get("crop_size", {}).get("height", 224))
        self.mean = np.array(cfg.get("image_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.array(cfg.get("image_std", [0.229, 0.224, 0.225]), dtype=np.float32)

    @property
    def available(self) -> bool:
        return self.model_path.exists()

    def load(self):
        if self._session is None:
            with self._lock:
                if self._session is None:
                    if not self.model_path.exists():
                        raise SimilarityUnavailableError("embedding model not installed")
                    import onnxruntime as ort

                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = 1
                    opts.inter_op_num_threads = 1
                    self._session = ort.InferenceSession(str(self.model_path), opts, providers=["CPUExecutionProvider"])
                    log.info("Embedding model loaded from %s", self.model_path.name)
        return self._session

    def preprocess(self, img: Image.Image) -> np.ndarray:
        img = img.convert("RGB")
        w, h = img.size
        scale = self.short_edge / min(w, h)
        img = img.resize((max(self.crop, round(w * scale)), max(self.crop, round(h * scale))), Image.BICUBIC)
        w, h = img.size
        left, top = (w - self.crop) // 2, (h - self.crop) // 2
        img = img.crop((left, top, left + self.crop, top + self.crop))
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = (x - self.mean) / self.std
        return x.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    def embed(self, img: Image.Image) -> np.ndarray:
        """Return an L2-normalised 384-d embedding (the DINOv2 [CLS] token)."""
        session = self.load()
        hidden = session.run(None, {session.get_inputs()[0].name: self.preprocess(img)})[0]
        vec = hidden[0, 0].astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-12)


class VectorIndex:
    """Exact inner-product index over normalised vectors (FAISS if available)."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.backend = "numpy"
        self._faiss = None
        try:
            import faiss

            index = faiss.IndexFlatIP(self.vectors.shape[1])
            index.add(self.vectors)
            self._faiss, self.backend = index, "faiss"
        except Exception:  # noqa: BLE001 - faiss missing or blocked: NumPy gives identical results
            pass

    @classmethod
    def load(cls, index_dir: Path) -> "VectorIndex":
        obj = cls(np.load(index_dir / "embeddings.npy"))
        try:  # use the saved FAISS index when faiss can read it and it matches
            import faiss

            saved = faiss.read_index(str(index_dir / "faiss.index"))
            if saved.ntotal == len(obj.vectors) and saved.d == obj.vectors.shape[1]:
                obj._faiss, obj.backend = saved, "faiss"
        except Exception:  # noqa: BLE001 - keep the in-memory FAISS / NumPy index built above
            pass
        return obj

    def __len__(self) -> int:
        return len(self.vectors)

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, len(self))
        q = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        if self._faiss is not None:
            scores, ids = self._faiss.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        scores = self.vectors @ q[0]
        top = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in top]


class EmbeddingCache:
    """A few recent embeddings keyed by the SHA-256 of the uploaded bytes.

    The main app calls /verify and then /search with the same file; the cache
    lets the second call reuse the first embedding instead of running DINOv2
    twice. Only vectors are kept (never images), and only for a handful of uploads."""

    def __init__(self, size: int = 16):
        self.size = size
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> np.ndarray | None:
        with self._lock:
            vec = self._data.get(key)
            if vec is not None:
                self._data.move_to_end(key)
            return vec

    def put(self, key: str, vec: np.ndarray) -> None:
        with self._lock:
            self._data[key] = vec
            self._data.move_to_end(key)
            while len(self._data) > self.size:
                self._data.popitem(last=False)


class SimilaritySearch:
    """Loads the encoder, the reference index and the leaf validator once."""

    def __init__(self, model_path: Path, index_dir: Path):
        self.encoder = ImageEncoder(model_path)
        self.index_dir = Path(index_dir)
        self.index: VectorIndex | None = None
        self.items: list[dict] = []
        self._thumbs: dict[str, Path] = {}
        self.info: dict = {}
        self.status = "not_loaded"
        self.validator = LeafValidator(self.index_dir / "leaf_validator.json")
        self.cache = EmbeddingCache()

    @property
    def available(self) -> bool:
        return self.status == "ready"

    @property
    def validator_available(self) -> bool:
        return self.available and self.validator.loaded

    def embedding(self, img: Image.Image, data: bytes | None = None) -> tuple[np.ndarray, bool]:
        """Embed an image, reusing the cached vector for identical upload bytes."""
        key = hashlib.sha256(data).hexdigest() if data is not None else None
        if key is not None:
            vec = self.cache.get(key)
            if vec is not None:
                return vec, True
        vec = self.encoder.embed(img)
        if key is not None:
            self.cache.put(key, vec)
        return vec, False

    def verify(self, img: Image.Image, data: bytes | None = None) -> dict:
        """Is this a cauliflower leaf? Raises SimilarityUnavailableError if the check cannot run."""
        if not self.validator_available:
            raise SimilarityUnavailableError("leaf validator unavailable")
        t0 = time.perf_counter()
        vec, cached = self.embedding(img, data)
        out = self.validator.check(vec)
        out["model"] = self.validator.info.get("model", "DINOv2-small + logistic regression")
        out["timing_ms"] = {"embedding": 0.0 if cached else round((time.perf_counter() - t0) * 1000, 1)}
        return out

    def load(self) -> None:
        try:
            if not self.encoder.available:
                raise SimilarityUnavailableError("embedding model not installed")
            meta = json.loads((self.index_dir / "metadata.json").read_text(encoding="utf-8"))
            index = VectorIndex.load(self.index_dir)
            if len(index) != len(meta["items"]) or index.vectors.shape[1] != EMBEDDING_DIM:
                raise SimilarityUnavailableError("index and metadata do not match")
            self.encoder.load()
            thumbs = {}
            for it in meta["items"]:  # every id must be safe and have its thumbnail
                if not _ID.match(str(it.get("id", ""))) or not isinstance(it.get("label"), str):
                    raise SimilarityUnavailableError("invalid metadata entry")
                path = self.index_dir / "thumbs" / f"{it['id']}.jpg"
                if not path.is_file():
                    raise SimilarityUnavailableError(f"thumbnail missing for {it['id']}")
                thumbs[it["id"]] = path
            if len(thumbs) != len(meta["items"]):
                raise SimilarityUnavailableError("duplicate ids in metadata")
            self.index, self.items, self.info = index, meta["items"], meta.get("info", {})
            self._thumbs = thumbs
            try:
                self.validator.load()
            except ValidatorUnavailableError as exc:  # search still works; /verify reports 503
                log.warning("Leaf validator disabled: %s", exc)
            self.status = "ready"
            log.info("Similarity index loaded: %d reference images (%s)", len(index), index.backend)
        except (FileNotFoundError, SimilarityUnavailableError, KeyError, ValueError) as exc:
            self.status = "unavailable"
            log.warning("Similarity search disabled: %s", exc)

    def thumbnail_data(self, item_id: str) -> str:
        """Inline JPEG thumbnail of an indexed reference image, looked up by id only
        (ids are validated at load; there is no client-supplied path anywhere)."""
        path = self._thumbs[item_id]  # KeyError for anything not in the index
        return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()

    def search(self, img: Image.Image, k: int = 5, data: bytes | None = None) -> dict:
        if not self.available:
            raise SimilarityUnavailableError("similarity search unavailable")
        t0 = time.perf_counter()
        vec, cached = self.embedding(img, data)
        t1 = t0 if cached else time.perf_counter()
        hits = self.index.search(vec, k)
        t2 = time.perf_counter()
        results = [
            {
                "id": self.items[i]["id"],
                "label": self.items[i]["label"],
                "similarity": round(max(score, 0.0), 4),
                "thumbnail": self.thumbnail_data(self.items[i]["id"]),
            }
            for i, score in hits
        ]
        return {
            "results": results,
            "model": "DINOv2-small (ONNX)",
            "backend": self.index.backend,
            "reference_images": len(self.index),
            "timing_ms": {"embedding": round((t1 - t0) * 1000, 1), "search": round((t2 - t1) * 1000, 2)},
        }
