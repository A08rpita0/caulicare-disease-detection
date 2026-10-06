"""Sanity-check the similarity search and the affected-area estimate on the test split.

    python scripts/evaluate_visual_features.py --dataset "data/raw/Original Dataset"

Writes reports/visual_features.json. The test split is not in the reference
index, so retrieval is measured on images the index has not seen. These are
sanity checks, not validations: the dataset has no lesion masks or severity
labels, and different photos of the same plant can fall in different splits.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.predictor import CLASS_NAMES  # noqa: E402
from services.severity import SeverityError, band_for, estimate  # noqa: E402
from similarity_service.engine import SimilaritySearch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "data" / "raw" / "Original Dataset"))
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    dataset = Path(args.dataset)
    test = json.loads((ROOT / "data" / "splits.json").read_text())["splits"]["test"]

    sim = SimilaritySearch(ROOT / "similarity_service" / "models" / "dinov2-small.onnx",
                           ROOT / "similarity_service" / "index")
    sim.load()
    if not sim.available:
        print("Similarity index unavailable; run the download and build scripts first.", file=sys.stderr)
        return 1

    top1 = majority = 0
    per_class_top1, per_class_n = Counter(), Counter()
    top_sims, emb_ms, search_ms, sev_ms = [], [], [], []
    severity = defaultdict(list)
    not_estimated = defaultdict(Counter)
    for e in test:
        label = CLASS_NAMES[e["label"]]
        img = Image.open(dataset / e["path"]).convert("RGB")
        res = sim.search(img, args.k)
        labels = [r["label"] for r in res["results"]]
        top1 += labels[0] == label
        majority += Counter(labels).most_common(1)[0][0] == label
        per_class_n[label] += 1
        per_class_top1[label] += labels[0] == label
        top_sims.append(res["results"][0]["similarity"])
        emb_ms.append(res["timing_ms"]["embedding"])
        search_ms.append(res["timing_ms"]["search"])
        t = time.perf_counter()
        try:
            severity[label].append(estimate(img).affected_percent)
        except SeverityError as exc:
            not_estimated[label]["cauliflower head" if "head" in str(exc) else "too little leaf"] += 1
        sev_ms.append((time.perf_counter() - t) * 1000)

    n = len(test)
    report = {
        "test_images": n,
        "reference_images": len(sim.index),
        "index_backend": sim.index.backend,
        "similarity": {
            "top1_label_agreement": round(top1 / n, 4),
            "top1_correct": top1,
            "top5_majority_label_agreement": round(majority / n, 4),
            "per_class_top1": {c: f"{per_class_top1[c]}/{per_class_n[c]}" for c in CLASS_NAMES},
            "top1_similarity_median": round(float(np.median(top_sims)), 3),
            "caveat": "Many test photos have other photos of the same plant in the reference set, "
                      "so label agreement overstates performance on new plants.",
        },
        "severity_estimate": {
            c: {
                "estimated": len(severity[c]),
                "not_estimated": dict(not_estimated[c]),
                "median_percent": round(float(np.median(severity[c])), 1) if severity[c] else None,
                "iqr_percent": [round(float(np.percentile(severity[c], q)), 1) for q in (25, 75)] if severity[c] else None,
                "bands": dict(Counter(band_for(v) for v in severity[c])),
            }
            for c in CLASS_NAMES
        },
        "latency_ms_median": {
            "embedding": round(float(np.median(emb_ms)), 1),
            "vector_search": round(float(np.median(search_ms)), 3),
            "severity": round(float(np.median(sev_ms)), 1),
        },
    }
    out = ROOT / "reports" / "visual_features.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
