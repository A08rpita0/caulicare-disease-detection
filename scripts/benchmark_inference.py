"""Benchmark CauliCare inference: Keras vs ONNX Runtime, single and batched, and the API end to end.

    python scripts/benchmark_inference.py                      # synthetic photos, writes reports/benchmark.json
    python scripts/benchmark_inference.py --dataset "data/raw/Original Dataset"   # also checks accuracy parity
                                                                                  # on the 99 held-out test images
Options: --runs N (timed runs per block, default 40), --rounds N (blocks per measurement, default 5),
--threads N (CPU threads for both runtimes, default 1 as in the Docker/Render configuration).

What is measured
  * model files: size on disk
  * parity: largest probability and Grad-CAM difference between Keras and ONNX, and top-class agreement
  * forward pass: latency for one 256x256 image (model only, preprocessing excluded)
  * batching: images per second for batches of 1, 4, 8 and 16
  * predictor: one decoded photo to a prediction, with and without Grad-CAM, on each backend
  * API: wall-clock time of POST /api/predict in each mode, through the Flask test client. The
    cauliflower-leaf check and similarity search are stubbed (they are network calls to another
    service), so these numbers are the main service's own work: decode, quality gate, preprocessing,
    CNN, Grad-CAM, affected-area estimate.
  * whole application, TensorFlow vs TensorFlow-free (INFERENCE_BACKEND=onnx), each in a fresh process:
    start-up time, resident memory after 12 full analyses (Grad-CAM included) and request latency.

Latency depends on the machine: compare the rows with each other, not with other computers' numbers.
Each latency is the median of the quietest of several blocks of runs (other programs on a laptop
otherwise add noise); treat differences under ~20% as noise.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROUNDS = 5


def stats(ms: list[float]) -> dict:
    s = sorted(ms)
    return {"mean": round(statistics.fmean(s), 2), "p50": round(s[len(s) // 2], 2),
            "p95": round(s[min(len(s) - 1, int(0.95 * len(s)))], 2), "min": round(s[0], 2), "runs": len(s)}


def timed(fn, runs: int, warmup: int = 5) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000)
    return out


def timed_best(fn, runs: int, rounds: int | None = None, warmup: int = 5) -> list[float]:
    """`rounds` blocks of `runs` timings; keep the block with the lowest median. A laptop has other
    programs running, which show up as slow blocks: the quietest block is the closest to the code's own cost."""
    blocks = [timed(fn, runs, warmup if i == 0 else 2) for i in range(rounds or ROUNDS)]
    return min(blocks, key=lambda b: sorted(b)[len(b) // 2])


def synthetic_photo(seed: int, size=(1024, 768)) -> bytes:
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    w, h = size
    # Plant-coloured noise plus a smooth pattern, so the frame is not "blank" after downscaling.
    pattern = 35 * np.sin(np.arange(w) / 37.0)[None, :] * np.sin(np.arange(h) / 53.0)[:, None]
    arr = np.stack([60 + pattern + rng.normal(0, 14, (h, w)), 140 + pattern + rng.normal(0, 14, (h, w)),
                    70 + pattern + rng.normal(0, 14, (h, w))], -1)
    buf = io.BytesIO()
    Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def make_stub():
    """Stands in for the similarity service (a network call to another process)."""
    from services.similarity_client import SimilarityClient

    class Stub(SimilarityClient):
        def __init__(self):
            super().__init__("")

        def verify(self, data, filename="image", mimetype="application/octet-stream"):
            return {"checked": True, "is_cauliflower_leaf": True, "score": 0.99, "threshold": 0.717}

        def search(self, data, filename="image", mimetype="application/octet-stream"):
            return {"available": True, "results": [], "timing_ms": {}}

    return Stub()


def rss_mb():
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 1e6)
    except ImportError:
        return None


# ----------------------------------------------------------------- fresh-process probes
def probe_app(kind: str, threads: int) -> None:
    """The whole application in this process: `kind` is "tensorflow" (Keras) or "onnx" (TensorFlow never imported)."""
    t0 = time.perf_counter()
    if kind == "onnx":
        sys.modules["tensorflow"] = None  # proves the process never needs it: any import fails
    from app import create_app
    from config import TestConfig
    from services.predictor import Predictor

    class Cfg(TestConfig):
        RATE_LIMIT_PREDICT = "1000000/minute"

    onnx = ROOT / "models" / "model1.onnx"
    pred = Predictor(ROOT / "models" / "model1.h5", onnx_path=onnx if kind == "onnx" else None,
                     onnx_threads=threads, backend="onnx" if kind == "onnx" else "tensorflow")
    client = create_app(Cfg, predictor=pred, similarity=make_stub()).test_client()
    pred.load()
    ready = time.perf_counter() - t0
    photos = [synthetic_photo(300 + i) for i in range(6)]
    lat = []
    for i in range(12):
        t = time.perf_counter()
        r = client.post("/api/predict", data={"image": (io.BytesIO(photos[i % 6]), "leaf.jpg")},
                        content_type="multipart/form-data")
        lat.append((time.perf_counter() - t) * 1000)
        assert r.status_code == 200 and "explanation" in r.json["prediction"], r.json
    print(json.dumps({"start_seconds": round(ready, 2), "rss_mb_after_12_full_analyses": rss_mb(),
                      "first_request_ms": round(lat[0]), "full_request_ms_p50": round(sorted(lat[2:])[len(lat[2:]) // 2], 1),
                      "tensorflow_imported": any(m.startswith("tensorflow") and v is not None for m, v in sys.modules.items()),
                      "gradcam_backend": pred.gradcam_backend}))


def run_probe(kind: str, threads: int) -> dict:
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "TF_NUM_INTRAOP_THREADS": str(threads),
           "TF_NUM_INTEROP_THREADS": "1"}
    out = subprocess.run([sys.executable, "-W", "ignore", __file__, "--probe", kind, "--threads", str(threads)],
                         capture_output=True, text=True, env=env, timeout=900)
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    return json.loads(line[-1]) if line else {"error": out.stderr[-400:]}


# ----------------------------------------------------------------- main benchmark
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=5, help="blocks of runs per measurement; the quietest is reported")
    ap.add_argument("--dataset", default=os.getenv("CAULICARE_DATASET_DIR", ""))
    ap.add_argument("--out", default=str(ROOT / "reports" / "benchmark.json"))
    ap.add_argument("--probe", choices=["tensorflow", "onnx"], help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.probe:
        probe_app(args.probe, args.threads)
        return 0

    global ROUNDS
    ROUNDS = max(1, args.rounds)
    try:  # fewer interruptions from other programs
        import psutil

        psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS if os.name == "nt" else -5)
    except Exception:  # noqa: BLE001 - optional
        pass
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(args.threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    import numpy as np
    import onnxruntime as ort
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from PIL import Image

    from services.predictor import Predictor

    keras_path, onnx_path, int8_path = ROOT / "models" / "model1.h5", ROOT / "models" / "model1.onnx", ROOT / "models" / "model1.int8.onnx"
    if not onnx_path.exists():
        print("models/model1.onnx is missing: run python scripts/export_onnx.py first.", file=sys.stderr)
        return 1

    report: dict = {
        "machine": {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
                    "tensorflow": tf.__version__, "onnxruntime": ort.__version__, "threads": args.threads,
                    "rounds": ROUNDS, "runs_per_round": args.runs},
        "files_kb": {p.name: round(p.stat().st_size / 1024) for p in
                     (keras_path, onnx_path, onnx_path.with_name("model1.gradcam.npz"), int8_path) if p.exists()},
    }
    print("model files (KB):", report["files_kb"])

    model = tf.keras.models.load_model(keras_path, compile=False)

    def session(path):
        o = ort.SessionOptions()
        o.intra_op_num_threads, o.inter_op_num_threads = args.threads, 1
        return ort.InferenceSession(str(path), sess_options=o, providers=["CPUExecutionProvider"])

    sessions = {"onnx_fp32": session(onnx_path)}
    if int8_path.exists():
        sessions["onnx_int8_dynamic"] = session(int8_path)

    def run_onnx(s, x):
        return s.run(["probabilities"], {"image": x})[0]

    # -- images: real test photos when available, otherwise synthetic
    rng = np.random.default_rng(0)
    labels = None
    if args.dataset and Path(args.dataset).exists():
        from services.predictor import preprocess_image

        split = json.loads((ROOT / "data" / "splits.json").read_text())["splits"]["test"]
        arrays = [preprocess_image(Image.open(Path(args.dataset) / e["path"])) for e in split]
        labels = np.array([e["label"] for e in split])
        report["images"] = f"{len(arrays)} held-out test photos"
    else:
        arrays = [rng.uniform(0, 255, (1, 256, 256, 3)).astype("float32") for _ in range(32)]
        report["images"] = "32 synthetic images (latency does not depend on pixel content)"
    x1 = arrays[0]

    # -- parity
    allx = np.concatenate(arrays)
    keras_p = np.concatenate([model(allx[i:i + 16], training=False).numpy() for i in range(0, len(allx), 16)])
    parity = {}
    for name, s in sessions.items():
        p = np.concatenate([run_onnx(s, allx[i:i + 16]) for i in range(0, len(allx), 16)])
        parity[name] = {"max_abs_probability_difference": float(f"{np.abs(keras_p - p).max():.3g}"),
                        "same_top_class": f"{int((keras_p.argmax(1) == p.argmax(1)).sum())}/{len(allx)}"}
        if labels is not None:
            parity[name]["accuracy"] = round(float((p.argmax(1) == labels).mean()), 4)
    if labels is not None:
        parity["keras_accuracy"] = round(float((keras_p.argmax(1) == labels).mean()), 4)
    pred_tf = Predictor(keras_path, onnx_threads=args.threads, backend="tensorflow")
    pred = Predictor(keras_path, onnx_path=onnx_path, onnx_threads=args.threads, backend="auto")
    pred_tf.load()
    pred.load()
    worst = 0.0
    for a in arrays[:12]:
        _, cam_k = pred_tf._forward(pred_tf._model, a, True)
        _, cam_o = pred._cam(pred._onnx, pred._gradcam, a)
        worst = max(worst, float(np.abs(cam_k - cam_o).max()))
    parity["gradcam_max_abs_difference_onnx_vs_keras"] = float(f"{worst:.3g}")
    report["parity"] = parity
    print("parity:", json.dumps(parity))

    # -- single-image forward pass
    forward = {"keras": stats(timed_best(lambda: model(x1, training=False).numpy(), args.runs))}
    for name, s in sessions.items():
        forward[name] = stats(timed_best(lambda s=s: run_onnx(s, x1), args.runs))
    report["forward_pass_ms"] = forward
    for name, v in forward.items():
        print(f"forward {name:18s} p50 {v['p50']:7.2f} ms   p95 {v['p95']:7.2f} ms")

    # -- batching
    batching = {}
    for n in (1, 4, 8, 16):
        xb = np.concatenate((arrays * 16)[:n])
        row = {}
        k = stats(timed_best(lambda xb=xb: model(xb, training=False).numpy(), max(10, args.runs // 2)))
        row["keras_images_per_second"] = round(n / (k["p50"] / 1000), 1)
        for name, s in sessions.items():
            o = stats(timed_best(lambda s=s, xb=xb: run_onnx(s, xb), max(10, args.runs // 2)))
            row[f"{name}_images_per_second"] = round(n / (o["p50"] / 1000), 1)
        batching[f"batch_{n}"] = row
    report["batching"] = batching
    print("batching:", json.dumps(batching))

    # -- one decoded photo to a prediction
    img = Image.open(io.BytesIO(synthetic_photo(1))).convert("RGB")
    report["predictor_ms"] = {
        "gradcam_keras": stats(timed_best(lambda: pred_tf.predict(img, explain=True), max(10, args.runs // 2))),
        "gradcam_onnx": stats(timed_best(lambda: pred.predict(img, explain=True), max(10, args.runs // 2))),
        "no_gradcam_keras": stats(timed_best(lambda: pred_tf.predict(img, explain=False), args.runs)),
        "no_gradcam_onnx": stats(timed_best(lambda: pred.predict(img, explain=False), args.runs)),
    }
    print("predictor p50:", {k: v["p50"] for k, v in report["predictor_ms"].items()})

    # -- API end to end
    report["api_ms"] = api_benchmark(pred_tf, pred, args.runs)
    print("api p50:", {k: v["p50"] for k, v in report["api_ms"].items() if isinstance(v, dict) and "p50" in v})

    # -- whole application in a fresh process, with and without TensorFlow
    report["application_process"] = {"tensorflow": run_probe("tensorflow", args.threads),
                                     "onnx_without_tensorflow": run_probe("onnx", args.threads)}
    print("application:", json.dumps(report["application_process"]))

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")
    return 0


def api_benchmark(pred_tf, pred_onnx, runs: int) -> dict:
    from app import create_app
    from config import TestConfig

    class NoCache(TestConfig):  # every request does the full work
        RATE_LIMIT_PREDICT = "1000000/minute"

    class WithCache(NoCache):
        PREDICTION_CACHE_SIZE = 64

    keras = create_app(NoCache, predictor=pred_tf, similarity=make_stub()).test_client()
    cold = create_app(NoCache, predictor=pred_onnx, similarity=make_stub()).test_client()
    warm = create_app(WithCache, predictor=pred_onnx, similarity=make_stub()).test_client()
    photos = [synthetic_photo(100 + i) for i in range(12)]

    def call(client, data, query=None):
        r = client.post("/api/predict", data={"image": (io.BytesIO(data), "leaf.jpg")}, query_string=query or {},
                        content_type="multipart/form-data")
        if r.status_code != 200:
            raise SystemExit(f"API benchmark request failed: {r.status_code} {r.json}")
        return r

    def series(client, query):
        i = iter(range(10**6))
        return stats(timed_best(lambda: call(client, photos[next(i) % len(photos)], query), runs, rounds=2))

    out = {
        "full_gradcam_keras": series(keras, {}),
        "full_gradcam_onnx": series(cold, {}),
        "no_gradcam_onnx": series(cold, {"explain": "0"}),
        "compact_onnx": series(cold, {"compact": "1"}),
    }
    call(warm, photos[0], {"compact": "1"})
    out["repeat_upload_compact_cache_hit"] = stats(timed(lambda: call(warm, photos[0], {"compact": "1"}), runs))
    call(warm, photos[0], {})
    out["repeat_upload_full_cache_hit"] = stats(timed(lambda: call(warm, photos[0], {}), runs))

    # eight images: one batched request vs eight compact requests (no cache in either)
    eight = photos[:8]
    reps = max(8, runs // 3)

    def batch_once():
        files = {"images": [(io.BytesIO(b), f"{i}.jpg") for i, b in enumerate(eight)]}
        r = cold.post("/api/predict/batch", data=files, content_type="multipart/form-data")
        assert r.status_code == 200 and r.json["count"] == 8 and r.json["rejected"] == 0, r.json

    out["eight_images_one_batch_request"] = stats(timed_best(batch_once, reps, rounds=2, warmup=2))
    out["eight_images_eight_compact_requests"] = stats(timed_best(
        lambda: [call(cold, b, {"compact": "1"}) for b in eight], reps, rounds=2, warmup=2))
    out["example_stage_timing_ms_compact"] = call(cold, photos[1], {"compact": "1"}).json["timing_ms"]
    out["example_stage_timing_ms_full"] = call(cold, photos[1], {}).json["timing_ms"]
    out["note"] = ("Leaf check and similarity search are stubbed; in production the leaf check adds one network "
                   "round trip to the similarity service.")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
