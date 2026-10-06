"""Fit a softmax temperature on the VALIDATION split (never test) and report ECE.

T > 1 softens over-confident scores; argmax (the predicted class) is unchanged.
The result is recorded in models/model_metadata.json. It did not reduce test ECE
(0.102 -> 0.102), so the app serves the raw softmax scores.
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402

from common import MODELS, REPORTS, SPLIT_FILE  # noqa: E402
from evaluate import probs_for  # noqa: E402


def soften(p, t):
    z = np.log(np.clip(p, 1e-12, 1)) / t
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def ece(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1)
    total = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def main():
    import tensorflow as tf

    model = tf.keras.models.load_model(MODELS / "model1.h5", compile=False)
    split = json.loads(SPLIT_FILE.read_text())
    out = {}
    data = {}
    for name in ("val", "test"):
        entries = split["splits"][name]
        data[name] = (probs_for(model, entries), np.array([e["label"] for e in entries]))
    pv, yv = data["val"]
    grid = np.exp(np.linspace(np.log(0.5), np.log(10), 400))
    t = float(grid[np.argmin([nll(soften(pv, g), yv) for g in grid])])
    pt, yt = data["test"]
    out = {"temperature": round(t, 3), "fit_on": "validation",
           "val_nll_before": nll(pv, yv), "val_nll_after": nll(soften(pv, t), yv),
           "test_ece_before": ece(pt, yt), "test_ece_after": ece(soften(pt, t), yt),
           "test_mean_conf_before": float(pt.max(1).mean()), "test_mean_conf_after": float(soften(pt, t).max(1).mean()),
           "test_accuracy": float((pt.argmax(1) == yt).mean())}
    (REPORTS / "calibration.json").write_text(json.dumps(out, indent=1))
    meta_path = MODELS / "model_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["temperature"] = out["temperature"]
    meta["calibration"] = {k: v for k, v in out.items() if k != "temperature"}
    meta_path.write_text(json.dumps(meta, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
