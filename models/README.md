# Model artifacts

- `model1.h5` — the trained CNN used by the application (2.3 MB, committed). Used for Grad-CAM and as the reference model.
- `model1.onnx` — the same network exported for ONNX Runtime (0.7 MB, committed). Used for plain predictions
  (compact API, batches, `explain=0`). Create it again with `python scripts/export_onnx.py`; the app compares it
  with the `.h5` model when it starts and ignores it if they differ.
- `model1.gradcam.npz` — the few weights needed to compute Grad-CAM from the ONNX model's activations.
- An optional dynamic int8 copy can be written with `python scripts/export_onnx.py --quantize`. It was not faster
  (see `reports/benchmark.json`), so it is not used and not committed.
- `model_metadata.json` — input size, class order, parameter count and measured metrics.

To reproduce the model from the original dataset, see `training/README.md`.
