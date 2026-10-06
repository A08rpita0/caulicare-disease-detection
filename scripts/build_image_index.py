"""Build the reference index for visual similarity search.

    python scripts/build_image_index.py --dataset "data/raw/Original Dataset"

Steps: load the reference images listed in data/splits.json (train + validation
by default; the test split is kept out so it can be used to evaluate retrieval),
validate that each decodes, embed it with DINOv2-small, L2-normalise, build an
exact FAISS inner-product index, and write to similarity_service/index/:

    embeddings.npy   float32 [N, 384], row i <-> metadata item i
    faiss.index      FAISS IndexFlatIP (written when the faiss package is available)
    metadata.json    id, class label, split and dataset-relative source path per vector
    thumbs/<id>.jpg  small thumbnails shown on the result page

The index is committed: the images come from the VegNet dataset (CC BY 4.0),
see similarity_service/index/ATTRIBUTION.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.predictor import CLASS_NAMES  # noqa: E402
from similarity_service.engine import EMBEDDING_DIM, ImageEncoder  # noqa: E402

THUMB_SIZE = 200


def slug(label: str) -> str:
    return label.lower().replace(" ", "-")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(ROOT / "data" / "raw" / "Original Dataset"))
    ap.add_argument("--splits", default="train,val", help="comma-separated splits from data/splits.json")
    ap.add_argument("--model", default=str(ROOT / "similarity_service" / "models" / "dinov2-small.onnx"))
    ap.add_argument("--out", default=str(ROOT / "similarity_service" / "index"))
    args = ap.parse_args()

    dataset, out = Path(args.dataset), Path(args.out)
    if not dataset.is_dir():
        print(f"Dataset folder not found: {dataset}", file=sys.stderr)
        return 1
    encoder = ImageEncoder(Path(args.model))
    if not encoder.available:
        print("Embedding model missing. Run: python similarity_service/download_model.py", file=sys.stderr)
        return 1

    split_data = json.loads((ROOT / "data" / "splits.json").read_text())["splits"]
    wanted = [s.strip() for s in args.splits.split(",") if s.strip()]
    (out / "thumbs").mkdir(parents=True, exist_ok=True)

    items, vectors, skipped = [], [], []
    counters: dict[str, int] = {}
    t0 = time.perf_counter()
    for split in wanted:
        for entry in split_data[split]:
            rel = entry["path"]
            label = CLASS_NAMES[entry["label"]]
            try:
                with Image.open(dataset / rel) as im:
                    im.load()
                    img = im.convert("RGB")
            except Exception as exc:  # noqa: BLE001 - report and skip unreadable files
                skipped.append(f"{rel}: {exc.__class__.__name__}")
                continue
            vec = encoder.embed(img)
            if vec.shape != (EMBEDDING_DIM,) or not np.isfinite(vec).all():
                skipped.append(f"{rel}: invalid embedding")
                continue
            counters[label] = counters.get(label, 0) + 1
            item_id = f"{slug(label)}-{counters[label]:03d}"
            thumb = img.copy()
            thumb.thumbnail((THUMB_SIZE, THUMB_SIZE))
            thumb.save(out / "thumbs" / f"{item_id}.jpg", quality=82)
            items.append({"id": item_id, "label": label, "split": split, "source": rel})
            vectors.append(vec)

    emb = np.stack(vectors).astype(np.float32)
    np.save(out / "embeddings.npy", emb)
    backend = "numpy (faiss not available here; the app builds the index in memory)"
    try:
        import faiss

        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(emb)
        faiss.write_index(index, str(out / "faiss.index"))
        backend = f"faiss {faiss.__version__} IndexFlatIP"
    except Exception:  # noqa: BLE001
        pass

    meta = {
        "info": {
            "model": "DINOv2-small (onnx-community/dinov2-small, ONNX)",
            "embedding": "[CLS] token of last_hidden_state, L2-normalised; inner product = cosine similarity",
            "splits": wanted,
            "count": len(items),
            "per_class": counters,
            "built": date.today().isoformat(),
            "index": backend,
        },
        "items": items,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"Indexed {len(items)} images in {time.perf_counter() - t0:.1f}s -> {out}")
    print(f"Per class: {counters}  Index: {backend}")
    if skipped:
        print(f"Skipped {len(skipped)}:", *skipped, sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
