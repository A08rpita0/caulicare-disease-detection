"""Export models/model1.h5 to ONNX and check that both models agree.

    python scripts/export_onnx.py                       # -> models/model1.onnx + models/model1.gradcam.npz
    python scripts/export_onnx.py --quantize            # also models/model1.int8.onnx (dynamic int8)

Run it again whenever model1.h5 changes: the ONNX file records the SHA-256 of the .h5 it was exported
from, and the TensorFlow-free server (INFERENCE_BACKEND=onnx) refuses a file that does not match.

The check feeds the same random images to Keras and ONNX Runtime and fails (exit 1) if the
probabilities or the Grad-CAM heat-maps differ by more than --tolerance, so a bad export is never
shipped silently.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "models" / "model1.h5"))
    ap.add_argument("--out", default=str(ROOT / "models" / "model1.onnx"))
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--quantize", action="store_true", help="also write a dynamic-int8 copy (<out>.int8.onnx)")
    args = ap.parse_args()

    import numpy as np
    import onnxruntime as ort
    import tensorflow as tf

    from services.gradcam_onnx import sidecar_path
    from services.onnx_export import export
    from services.predictor import Predictor

    model = tf.keras.models.load_model(args.model, compile=False)
    out = export(model, Path(args.out), source=Path(args.model))
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB; source {Path(args.model).stat().st_size / 1024:.0f} KB)")
    side = sidecar_path(out)
    print(f"Wrote {side} ({side.stat().st_size / 1024:.0f} KB)" if side.exists() else "No Grad-CAM parameters written.")

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 255, (16, 256, 256, 3)).astype(np.float32)
    expected = model(x, training=False).numpy()
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = session.run(None, {"image": x})[0]
    diff = float(np.abs(expected - got).max())
    same = bool((expected.argmax(1) == got.argmax(1)).all())
    print(f"max |keras - onnx| over 16 random images: {diff:.2e}; same top class: {same}")
    if diff > args.tolerance or not same:
        print("FAILED: the exported model does not match the Keras model.", file=sys.stderr)
        return 1

    if side.exists():
        ref = Predictor(Path(args.model), onnx_path=out, onnx_threads=1)
        ref.set_model(model)
        worst = 0.0
        for i in range(8):
            xi = x[i:i + 1]
            _, cam_keras = ref._forward(model, xi, True)
            _, cam_onnx = ref._cam(ref._onnx, ref._gradcam, xi)
            worst = max(worst, float(np.abs(cam_keras - cam_onnx).max()))
        print(f"max |keras - onnx| Grad-CAM heat-map over 8 random images: {worst:.2e}  ({ref.onnx_status})")
        if worst > args.tolerance:
            print("FAILED: the Grad-CAM computed from the ONNX activations does not match Keras.", file=sys.stderr)
            return 1

    if args.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        q = out.with_suffix(".int8.onnx")
        quantize_dynamic(str(out), str(q), weight_type=QuantType.QUInt8)
        qs = ort.InferenceSession(str(q), providers=["CPUExecutionProvider"])
        qdiff = float(np.abs(expected - qs.run(None, {"image": x})[0]).max())
        print(f"Wrote {q} ({q.stat().st_size / 1024:.0f} KB); max |keras - int8| = {qdiff:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
