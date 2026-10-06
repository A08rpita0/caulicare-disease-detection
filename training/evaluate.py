"""Evaluate a saved model in a fresh process on the held-out test split.

    python evaluate.py ../models/model_original_baseline.h5
    python evaluate.py ../models/model1.h5 --metadata   # also writes models/model_metadata.json

Images are loaded from the raw files through the *application's* preprocessing
(services.predictor.preprocess_image) so the evaluation exercises the deployed path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from common import CLASS_NAMES, DATASET_DIR, MODELS, REPORTS, ROOT, SPLIT_FILE  # noqa: E402

sys.path.insert(0, str(ROOT))
from services.predictor import preprocess_image  # noqa: E402


def probs_for(model, entries):
    out = []
    for e in entries:
        with Image.open(DATASET_DIR / e["path"]) as im:
            im.load()
            out.append(model(preprocess_image(im), training=False).numpy()[0])
    return np.array(out)


def main():
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, precision_recall_fscore_support

    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--metadata", action="store_true")
    args = ap.parse_args()
    path = Path(args.model).resolve()
    model = tf.keras.models.load_model(path, compile=False)
    split = json.loads(SPLIT_FILE.read_text())

    results = {"model": path.name, "params": model.count_params(), "input_shape": list(model.input_shape[1:]),
               "output_shape": list(model.output_shape[1:])}
    for name in ("train", "val", "test"):
        entries = split["splits"][name]
        y = np.array([e["label"] for e in entries])
        p = probs_for(model, entries)
        pred = p.argmax(1)
        results[f"{name}_accuracy"] = float(accuracy_score(y, pred))
        results[f"{name}_loss"] = float(log_loss(y, p, labels=list(range(len(CLASS_NAMES)))))
        if name == "test":
            prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=range(4), zero_division=0)
            mp, mr, mf, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
            wp, wr, wf, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
            results.update({
                "test_precision_macro": float(mp), "test_recall_macro": float(mr), "test_f1_macro": float(mf),
                "test_precision_weighted": float(wp), "test_recall_weighted": float(wr), "test_f1_weighted": float(wf),
                "per_class": {CLASS_NAMES[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(sup[i])} for i in range(4)},
                "confusion_matrix": {"labels": CLASS_NAMES, "rows_true_cols_pred": confusion_matrix(y, pred, labels=range(4)).tolist()},
                "test_prediction_counts": {CLASS_NAMES[i]: int((pred == i).sum()) for i in range(4)},
                "mean_top_probability_test": float(p.max(1).mean()),
            })
            print(classification_report(y, pred, target_names=CLASS_NAMES, digits=3, zero_division=0))
            samples = []
            for c in range(4):
                for k in [i for i in range(len(y)) if y[i] == c][:3]:
                    samples.append({"file": entries[k]["path"], "actual": CLASS_NAMES[y[k]], "predicted": CLASS_NAMES[pred[k]],
                                    "probabilities": {CLASS_NAMES[j]: round(float(p[k][j]), 4) for j in range(4)}})
            results["sample_predictions"] = samples

    REPORTS.mkdir(exist_ok=True)
    stem = path.stem
    (REPORTS / f"eval_{stem}.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v for k, v in results.items() if k not in ("sample_predictions", "per_class")}, indent=1))

    if args.metadata:
        train_info = {}
        for v in ("improved", "baseline"):
            f = REPORTS / f"train_{v}.json"
            if f.exists():
                info = json.loads(f.read_text())
                if info.get("artifact") in (path.name, "model_improved.h5" if path.name == "model1.h5" else None):
                    train_info = {k: info[k] for k in info if k != "history"}
                    break
        meta = {
            "model_type": "CNN (custom, trained from scratch; architecture from notebooks/Model.ipynb)",
            "input_size": [256, 256, 3], "classes": CLASS_NAMES, "parameters": results["params"],
            "preprocessing": "RGB, tf.image.resize bilinear 256x256, float32 0-255; Rescaling(1/255) inside model",
            "dataset": "Kaggle noamaanabdulazeem/cauliflower-dataset, 'Original Dataset' (656 images)",
            "split": split["counts"], "split_method": f"duplicate-aware (dHash<= {split['group_bits']} + MD5) class-stratified, seed {split['seed']}",
            "train_accuracy": results["train_accuracy"], "val_accuracy": results["val_accuracy"],
            "test_accuracy": results["test_accuracy"], "test_loss": results["test_loss"],
            "test_precision": results["test_precision_macro"], "test_recall": results["test_recall_macro"], "test_f1": results["test_f1_macro"],
            "metric_averaging": "macro", "training": train_info, "created": date.today().isoformat(),
            # Read by the web app (About page, disease pages); keep in sync with the report.
            "test_evaluation": {"source": f"reports/eval_{stem}.json", "images": split["counts"]["test"]["total"],
                                "per_class": results["per_class"], "confusion_matrix": results["confusion_matrix"]},
        }
        previous = MODELS / "model_metadata.json"
        if previous.exists() and "calibration" in (old := json.loads(previous.read_text())):
            meta["calibration"] = old["calibration"]  # written separately by calibrate.py
        (MODELS / "model_metadata.json").write_text(json.dumps(meta, indent=1))
        print("wrote models/model_metadata.json")


if __name__ == "__main__":
    main()
