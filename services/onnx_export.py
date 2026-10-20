"""Export the CauliCare Keras CNN to ONNX, without tf2onnx.

The network is a plain stack (Rescaling, Conv2D+ReLU, MaxPooling2D, Flatten, Dense), so
the ONNX graph is built directly from the Keras weights. This keeps the exporter
independent of the TensorFlow / Keras / tf2onnx version matrix. Training-only layers
(RandomFlip, RandomRotation) are inactive at inference and are dropped; Resizing is
dropped because the app already resizes to 256 x 256 before calling the model.

Graph contract (identical to the Keras model):
  input   "image"          float32 [batch, 256, 256, 3], raw 0-255 pixel values (NHWC)
  output  "probabilities"  float32 [batch, 4], softmax in CLASS_NAMES order
  output  "conv_features"  float32 [batch, H, W, C], last convolution after ReLU (for Grad-CAM)
  output  "hidden"         float32 [batch, D], Dense(ReLU) activations before the output layer

Next to the model, `<name>.gradcam.npz` holds the few weights services/gradcam_onnx.py needs to
compute Grad-CAM in NumPy. Both files record the SHA-256 of the Keras file they were exported
from, so a model that was retrained without re-exporting is detected.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from services.gradcam_onnx import sidecar_path

OPSET = 13
IR_VERSION = 8  # understood by every onnxruntime >= 1.14


class UnsupportedModelError(ValueError):
    """The Keras model has a layer this exporter does not know how to translate."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _flatten_layers(layers):
    for layer in layers:
        if layer.__class__.__name__ == "Sequential":
            yield from _flatten_layers(layer.layers)
        else:
            yield layer


def build_onnx(model, input_size: int = 256, source_sha256: str = ""):
    """Return (onnx.ModelProto, gradcam_parameters or None) for the Keras `model`.

    The Grad-CAM parameters are produced when the model ends in
    Conv2D -> MaxPooling2D -> Flatten -> Dense(ReLU) -> Dense, which is this project's network."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes, inits = [], []
    counter = iter(range(10_000))

    def name(prefix: str) -> str:
        return f"{prefix}_{next(counter)}"

    def add_init(prefix: str, array: np.ndarray) -> str:
        n = name(prefix)
        inits.append(numpy_helper.from_array(np.ascontiguousarray(array, dtype=np.float32), n))
        return n

    current = "image"
    nchw = False  # data layout of `current`
    layers = [layer for layer in _flatten_layers(model.layers)
              if layer.__class__.__name__ not in {"InputLayer", "Resizing", "RandomFlip", "RandomRotation", "Dropout"}]
    conv_out = hidden_out = None
    last_conv = pool_after_conv = None

    def to_nchw(x: str) -> str:
        out = name("nchw")
        nodes.append(helper.make_node("Transpose", [x], [out], perm=[0, 3, 1, 2]))
        return out

    def to_nhwc(x: str) -> str:
        out = name("nhwc")
        nodes.append(helper.make_node("Transpose", [x], [out], perm=[0, 2, 3, 1]))
        return out

    def activation(x: str, fn) -> str:
        fname = getattr(fn, "__name__", str(fn))
        if fname == "linear":
            return x
        if fname == "relu":
            out = name("relu")
            nodes.append(helper.make_node("Relu", [x], [out]))
            return out
        if fname == "softmax":
            out = name("softmax")
            nodes.append(helper.make_node("Softmax", [x], [out], axis=1))
            return out
        raise UnsupportedModelError(f"Unsupported activation: {fname}")

    for idx, layer in enumerate(layers):
        kind = layer.__class__.__name__
        if kind == "Rescaling":
            scale, offset = np.float32(layer.scale), np.float32(layer.offset)
            out = name("rescale")
            nodes.append(helper.make_node("Mul", [current, add_init("scale", np.array(scale))], [out]))
            current = out
            if offset != 0:
                out = name("offset")
                nodes.append(helper.make_node("Add", [current, add_init("offset", np.array(offset))], [out]))
                current = out
        elif kind == "Conv2D":
            if layer.padding != "valid" or tuple(layer.strides) != (1, 1) or tuple(layer.dilation_rate) != (1, 1):
                raise UnsupportedModelError("Only 'valid' stride-1 Conv2D layers are supported.")
            if not nchw:
                current, nchw = to_nchw(current), True
            kernel, bias = layer.get_weights()  # HWIO
            out = name("conv")
            nodes.append(helper.make_node(
                "Conv", [current, add_init("conv_w", kernel.transpose(3, 2, 0, 1)), add_init("conv_b", bias)], [out],
                kernel_shape=list(layer.kernel_size), strides=[1, 1], pads=[0, 0, 0, 0]))
            current = activation(out, layer.activation)
            conv_out, last_conv = current, layer
            nxt = layers[idx + 1] if idx + 1 < len(layers) else None
            pool_after_conv = nxt if nxt is not None and nxt.__class__.__name__ == "MaxPooling2D" else None
        elif kind == "MaxPooling2D":
            if layer.padding != "valid" or tuple(layer.pool_size) != tuple(layer.strides):
                raise UnsupportedModelError("Only 'valid' MaxPooling2D with strides == pool_size is supported.")
            if not nchw:
                current, nchw = to_nchw(current), True
            out = name("pool")
            nodes.append(helper.make_node("MaxPool", [current], [out], kernel_shape=list(layer.pool_size),
                                          strides=list(layer.strides), pads=[0, 0, 0, 0]))
            current = out
        elif kind == "Flatten":
            if nchw:  # Keras flattens NHWC; restore that order so Dense weights line up
                current, nchw = to_nhwc(current), False
            out = name("flatten")
            nodes.append(helper.make_node("Flatten", [current], [out], axis=1))
            current = out
        elif kind == "Dense":
            kernel, bias = layer.get_weights()
            out = name("dense")
            nodes.append(helper.make_node("Gemm", [current, add_init("dense_w", kernel), add_init("dense_b", bias)], [out]))
            current = activation(out, layer.activation)
            if getattr(layer.activation, "__name__", "") == "relu":
                hidden_out = current
        else:
            raise UnsupportedModelError(f"Unsupported layer: {kind}")

    if nchw:
        raise UnsupportedModelError("The model must end in dense layers.")
    nodes.append(helper.make_node("Identity", [current], ["probabilities"]))

    # Extra outputs for Grad-CAM, when the network has the expected tail.
    kinds = [layer.__class__.__name__ for layer in layers]
    dense = [layer for layer in layers if layer.__class__.__name__ == "Dense"]
    gradcam = None
    outputs = [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["batch", int(model.output_shape[-1])])]
    tail_ok = False
    if conv_out is not None and pool_after_conv is not None and hidden_out is not None and len(dense) >= 2:
        tail_ok = (kinds[layers.index(pool_after_conv) + 1:] == ["Flatten", "Dense", "Dense"]
                   and getattr(dense[-2].activation, "__name__", "") == "relu"
                   and getattr(dense[-1].activation, "__name__", "") == "softmax"
                   and pool_after_conv.pool_size[0] == pool_after_conv.pool_size[1])
    if tail_ok:
        nodes.append(helper.make_node("Transpose", [conv_out], ["conv_features"], perm=[0, 2, 3, 1]))
        nodes.append(helper.make_node("Identity", [hidden_out], ["hidden"]))
        outputs += [helper.make_tensor_value_info("conv_features", TensorProto.FLOAT, ["batch", None, None, None]),
                    helper.make_tensor_value_info("hidden", TensorProto.FLOAT, ["batch", int(dense[-2].units)])]
        gradcam = {"w_hidden": dense[-2].get_weights()[0], "w_out": dense[-1].get_weights()[0],
                   "pool": int(pool_after_conv.pool_size[0]), "layer": last_conv.name}

    graph = helper.make_graph(
        nodes, "caulicare_cnn",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", input_size, input_size, 3])],
        outputs, initializer=inits)
    proto = helper.make_model(graph, producer_name="caulicare", opset_imports=[helper.make_opsetid("", OPSET)])
    proto.ir_version = IR_VERSION
    if source_sha256:
        entry = proto.metadata_props.add()
        entry.key, entry.value = "source_sha256", source_sha256
    onnx.checker.check_model(proto)
    return proto, gradcam


def export(model, path: Path, source: Path | None = None, input_size: int = 256) -> Path:
    """Write `path` (the ONNX model) and `<name>.gradcam.npz` next to it.
    `source` is the Keras file the model was loaded from; its SHA-256 is recorded in both files."""
    import onnx

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(Path(source)) if source else ""
    proto, gradcam = build_onnx(model, input_size, digest)
    onnx.save(proto, str(path))
    side = sidecar_path(path)
    if gradcam is not None:
        np.savez(side, w_hidden=gradcam["w_hidden"].astype(np.float32), w_out=gradcam["w_out"].astype(np.float32),
                 pool=np.array(gradcam["pool"]), layer=np.array(gradcam["layer"]), source_sha256=np.array(digest))
    elif side.exists():
        side.unlink()  # never leave a sidecar that belongs to an older export
    return path
