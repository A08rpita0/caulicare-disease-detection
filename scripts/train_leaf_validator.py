"""Train and evaluate the cauliflower-leaf validator.

Input: the folder written by scripts/fetch_validator_data.py (manifest.csv +
images). Every image is embedded with the same DINOv2-small encoder the
similarity service uses, then:

  splits     VegNet           data/splits.json (train / val / test, unchanged)
             Durjoy et al.    Black Rot + Healthy photos grouped by perceptual hash,
                              60 / 20 / 20; all Insect Hole photos -> test only
                              (a leaf condition the validator never saw)
             negatives        PlantDoc, beans, pets, COCO: 60 / 20 / 20 per source
             unseen types     Oxford Flowers and synthetic documents/screenshots ->
                              test only, to measure image types absent from training

  models     logistic regression on the embedding (deployed)
             nearest-neighbour similarity to training leaves (compared)
             the colour/texture leaf heuristic in services/image_validation.py (compared)

  threshold  the highest score that keeps the validation false-rejection rate <= 1%;
             test data is never used to choose anything.

Outputs:
  similarity_service/index/leaf_validator.json   weights, bias, threshold (served)
  reports/leaf_validator.json                    every measured number

    python scripts/train_leaf_validator.py --data C:/caulicare-local-assets/validator_data
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.image_validation import assess_quality  # noqa: E402
from similarity_service.engine import ImageEncoder  # noqa: E402

TRAIN_NEG = ("plantdoc", "beans", "pets", "coco")
UNSEEN_NEG = ("flowers", "synthetic_docs")
MAX_VAL_FRR = 0.01
SEED = 42


# ----------------------------------------------------------------- data
def dhash(img: Image.Image) -> int:
    g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def group_split(paths: list[str], hashes: dict[str, int], rng: random.Random, max_dist: int = 6):
    """Near-identical photos (dHash distance <= max_dist) stay in the same split."""
    groups: list[list[str]] = []
    for p in paths:
        for g in groups:
            if bin(hashes[p] ^ hashes[g[0]]).count("1") <= max_dist:
                g.append(p)
                break
        else:
            groups.append([p])
    rng.shuffle(groups)
    n = sum(len(g) for g in groups)
    out, count = {}, 0
    for g in groups:
        split = "train" if count < 0.6 * n else "val" if count < 0.8 * n else "test"
        for p in g:
            out[p] = split
        count += len(g)
    return out


def load_rows(data: Path) -> list[dict]:
    with open(data / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
    return rows


def assign_splits(rows: list[dict], data: Path) -> None:
    rng = random.Random(SEED)
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    for source, items in by_source.items():
        if source == "vegnet":
            continue  # keeps data/splits.json
        if source in UNSEEN_NEG:
            for r in items:
                r["split"] = "test"
            continue
        if source == "durjoy_cauliflower":
            seen = [r for r in items if r["subclass"] != "2"]  # 2 = Insect Hole
            for r in items:
                if r["subclass"] == "2":
                    r["split"] = "test"
        else:
            seen = items
        hashes = {r["path"]: dhash(Image.open(data / r["path"])) for r in seen}
        split = group_split([r["path"] for r in seen], hashes, rng)
        for r in seen:
            r["split"] = split[r["path"]]


def embed_all(rows: list[dict], data: Path, encoder: ImageEncoder) -> np.ndarray:
    cache = data / "embeddings.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        known = dict(zip(z["paths"].tolist(), z["vectors"]))
    else:
        known = {}
    todo = [r for r in rows if r["path"] not in known]
    for i, r in enumerate(todo):
        known[r["path"]] = encoder.embed(Image.open(data / r["path"]).convert("RGB"))
        if i % 200 == 0:
            print(f"  embedded {i}/{len(todo)}", flush=True)
    if todo:
        paths = sorted(known)
        np.savez(cache, paths=np.array(paths), vectors=np.stack([known[p] for p in paths]))
    return np.stack([known[r["path"]] for r in rows]).astype(np.float32)


# --------------------------------------------------------------- metrics
def metrics(y: np.ndarray, accept: np.ndarray) -> dict:
    tp = int(np.sum(accept & (y == 1)))
    fn = int(np.sum(~accept & (y == 1)))
    fp = int(np.sum(accept & (y == 0)))
    tn = int(np.sum(~accept & (y == 0)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "positives": tp + fn, "negatives": fp + tn,
        "true_accept": tp, "false_reject": fn, "false_accept": fp, "true_reject": tn,
        "accuracy": round((tp + tn) / max(1, len(y)), 4),
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "false_acceptance_rate": round(fp / (fp + tn), 4) if fp + tn else None,
        "false_rejection_rate": round(fn / (tp + fn), 4) if tp + fn else None,
    }


def per_source(rows, y, accept, mask) -> dict:
    out = {}
    for source in sorted({r["source"] for r, m in zip(rows, mask) if m}):
        sm = mask & np.array([r["source"] == source for r in rows])
        n = int(sm.sum())
        acc = int((accept & sm).sum())
        out[source] = {"images": n, "accepted": acc, "rejected": n - acc,
                       "label": "cauliflower leaf" if y[sm][0] == 1 else "not cauliflower leaf"}
    return out


def pick_threshold(scores_pos_val: np.ndarray) -> float:
    """Highest threshold with validation false-rejection rate <= MAX_VAL_FRR."""
    s = np.sort(scores_pos_val)
    k = int(np.floor(MAX_VAL_FRR * len(s)))  # this many positives may fall below
    return float(s[k]) - 1e-6


# ----------------------------------------------------------------- models
def fit_logreg(X, y, C):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    clf.fit(X, y)
    return clf


def knn_scores(X, ref):
    return (X @ ref.T).max(axis=1)


# ------------------------------------------------------------ robustness
TRANSFORMS = {
    "rotate_90": lambda im: im.rotate(90, expand=True),
    "rotate_180": lambda im: im.rotate(180),
    "flip_horizontal": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
    "darker_x0.6": lambda im: ImageEnhance.Brightness(im).enhance(0.6),
    "brighter_x1.4": lambda im: ImageEnhance.Brightness(im).enhance(1.4),
    "low_saturation_x0.5": lambda im: ImageEnhance.Color(im).enhance(0.5),
    "centre_crop_60%": lambda im: im.crop((int(im.width * .2), int(im.height * .2), int(im.width * .8), int(im.height * .8))),
    "zoomed_out_on_soil": lambda im: _pad(im),
    "downscaled_to_224px": lambda im: im.resize((224, max(1, round(224 * im.height / im.width)))),
}


def _pad(im):
    canvas = Image.new("RGB", (im.width * 2, im.height * 2), (110, 84, 60))  # brown, soil-like border
    canvas.paste(im, (im.width // 2, im.height // 2))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--model", type=Path, default=ROOT / "similarity_service" / "models" / "dinov2-small.onnx")
    args = ap.parse_args()

    rows = load_rows(args.data)
    assign_splits(rows, args.data)
    encoder = ImageEncoder(args.model)
    print(f"embedding {len(rows)} images ...", flush=True)
    X = embed_all(rows, args.data, encoder)
    y = np.array([r["label"] for r in rows])
    split = np.array([r["split"] for r in rows])
    tr, va, te = split == "train", split == "val", split == "test"
    seen_te = te & np.array([r["source"] not in UNSEEN_NEG for r in rows])
    unseen_te = te & np.array([r["source"] in UNSEEN_NEG for r in rows])

    report: dict = {"date": date.today().isoformat(), "seed": SEED, "max_validation_frr": MAX_VAL_FRR,
                    "split_sizes": {}, "model_selection": {}}
    for name, m in (("train", tr), ("val", va), ("test", te)):
        report["split_sizes"][name] = {
            s: int(np.sum(m & np.array([r["source"] == s for r in rows]))) for s in sorted({r["source"] for r in rows})}

    # --- logistic regression: choose C on validation (lowest FAR at <= 1% FRR)
    best = None
    for C in (0.1, 0.3, 1.0, 3.0, 10.0):
        clf = fit_logreg(X[tr], y[tr], C)
        sv = clf.predict_proba(X[va])[:, 1]
        thr = pick_threshold(sv[y[va] == 1])
        far = float(np.mean(sv[y[va] == 0] >= thr))
        report["model_selection"][f"logreg_C={C}"] = {"val_threshold": round(thr, 4), "val_far": round(far, 4)}
        if best is None or far < best[0] - 1e-9:
            best = (far, C, clf, thr)
    _, C, clf, thr = best
    print(f"logistic regression: C={C} threshold={thr:.4f}")

    # --- nearest-neighbour baseline (one-class: needs no negatives)
    ref = X[tr & (y == 1)]
    knn_va = knn_scores(X[va], ref)
    knn_thr = pick_threshold(knn_va[y[va] == 1])

    def score_lr(Z):
        return clf.predict_proba(Z)[:, 1]

    s_all = score_lr(X)
    k_all = knn_scores(X, ref)
    accept_lr = s_all >= thr
    accept_knn = k_all >= knn_thr

    # --- colour/texture heuristic (the previous gate)
    heur = np.array([assess_quality(Image.open(args.data / r["path"]).convert("RGB")).looks_like_leaf for r in rows])

    def block(accept):
        return {
            "validation": metrics(y[va], accept[va]),
            "test_all": metrics(y[te], accept[te]),
            "test_seen_negative_types": metrics(y[seen_te], accept[seen_te]),
            "test_unseen_negative_types_only": metrics(y[unseen_te], accept[unseen_te]),
            "test_by_source": per_source(rows, y, accept, te),
        }

    report["deployed"] = {"model": "logistic regression on DINOv2-small CLS embedding",
                          "C": C, "threshold": round(thr, 6), **block(accept_lr)}
    report["compared"] = {
        "nearest_neighbour_similarity": {"threshold": round(knn_thr, 4), **block(accept_knn)},
        "colour_texture_heuristic": block(heur),
        "logreg_and_heuristic": block(accept_lr & heur),
    }

    # --- false accepts / rejects on test (for inspection)
    report["test_errors"] = {
        "false_accepts": [{"source": r["source"], "subclass": r["subclass"], "score": round(float(s), 4)}
                          for r, s, a, t in zip(rows, s_all, accept_lr, te) if t and a and r["label"] == 0],
        "false_rejects": [{"source": r["source"], "subclass": r["subclass"], "score": round(float(s), 4)}
                          for r, s, a, t in zip(rows, s_all, accept_lr, te) if t and not a and r["label"] == 1],
    }

    # --- robustness: positive test photos under common variations
    pos_te = [r for r, t in zip(rows, te) if t and r["label"] == 1]
    neg_te = [r for r, t in zip(rows, te) if t and r["label"] == 0]
    robust = {}
    for name, fn in TRANSFORMS.items():
        pv = np.stack([encoder.embed(fn(Image.open(args.data / r["path"]).convert("RGB"))) for r in pos_te])
        nv = np.stack([encoder.embed(fn(Image.open(args.data / r["path"]).convert("RGB"))) for r in neg_te[::4]])
        robust[name] = {
            "positives": len(pv), "false_rejection_rate": round(float(np.mean(score_lr(pv) < thr)), 4),
            "negatives_sampled": len(nv), "false_acceptance_rate": round(float(np.mean(score_lr(nv) >= thr)), 4),
        }
        print(f"  robustness {name}: {robust[name]}", flush=True)
    report["robustness_test"] = robust

    # --- ablation: trained on VegNet only -> how well does it accept the other dataset?
    veg_tr = tr & ((y == 0) | np.array([r["source"] == "vegnet" for r in rows]))
    clf_v = fit_logreg(X[veg_tr], y[veg_tr], C)
    veg_va = va & ((y == 0) | np.array([r["source"] == "vegnet" for r in rows]))
    sv = clf_v.predict_proba(X[veg_va])[:, 1]
    thr_v = pick_threshold(sv[y[veg_va] == 1])
    dj = te & np.array([r["source"] == "durjoy_cauliflower" for r in rows])
    report["ablation_vegnet_only_positives"] = {
        "note": "Same recipe, but no Durjoy photos in training: measures acceptance of cauliflower "
                "leaves from a dataset the validator never saw.",
        "durjoy_test_false_rejection_rate": round(float(np.mean(clf_v.predict_proba(X[dj])[:, 1] < thr_v)), 4),
        "durjoy_test_images": int(dj.sum()),
        "test_false_acceptance_rate": round(float(np.mean(clf_v.predict_proba(X[te & (y == 0)])[:, 1] >= thr_v)), 4),
    }

    out_model = {
        "model": "DINOv2-small CLS embedding + logistic regression",
        "version": date.today().isoformat(),
        "weights": [round(float(v), 7) for v in clf.coef_[0]],
        "bias": round(float(clf.intercept_[0]), 7),
        "threshold": round(thr, 6),
        "threshold_rule": f"highest score with validation false-rejection rate <= {MAX_VAL_FRR:.0%}",
        "trained_on": report["split_sizes"]["train"],
        "report": "reports/leaf_validator.json",
    }
    (ROOT / "similarity_service" / "index" / "leaf_validator.json").write_text(json.dumps(out_model, indent=1))
    (ROOT / "reports" / "leaf_validator.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report["deployed"][k] for k in ("validation", "test_all", "test_unseen_negative_types_only")},
                     indent=1))


if __name__ == "__main__":
    main()
