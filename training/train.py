"""Train the CauliCare CNN on the verified dataset with the leak-free split.

    python train.py --variant baseline   # faithful reconstruction of notebooks/Model.ipynb
    python train.py --variant improved   # same conv stack + dropout, class weights, early stopping

Preprocessing matches the notebook exactly: image_dataset_from_directory decodes
to RGB and resizes with tf.image.resize(bilinear, antialias=False) to 256x256,
keeping float values in [0, 255]; the model itself rescales by 1/255.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.keras import layers, models  # noqa: E402

from common import CLASS_NAMES, DATASET_DIR, IMAGE_SIZE, MODELS, REPORTS, ROOT, SEED, SPLIT_FILE  # noqa: E402

CACHE = ROOT / "data" / "processed" / "cache_256.npz"
BATCH_SIZE = 32  # notebook


def decode_resize(path: str) -> np.ndarray:
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    return tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE), method="bilinear").numpy()


def load_split_arrays():
    split = json.loads(SPLIT_FILE.read_text())
    if CACHE.exists():
        data = np.load(CACHE)
        return {k: (data[f"{k}_x"], data[f"{k}_y"]) for k in ("train", "val", "test")}, split
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out, arrays = {}, {}
    for name in ("train", "val", "test"):
        xs = np.stack([decode_resize(str(DATASET_DIR / e["path"])) for e in split["splits"][name]]).astype(np.float32)
        ys = np.array([e["label"] for e in split["splits"][name]], dtype=np.int32)
        out[name] = (xs, ys)
        arrays[f"{name}_x"], arrays[f"{name}_y"] = xs, ys
        print(f"decoded {name}: {xs.shape}")
    np.savez(CACHE, **arrays)
    return out, split


def build_model(variant: str) -> tf.keras.Model:
    """Architecture copied from the notebook; 'improved' only adds regularisation."""
    resize_and_rescale = tf.keras.Sequential(
        [layers.Resizing(IMAGE_SIZE, IMAGE_SIZE), layers.Rescaling(1.0 / 255)], name="resize_and_rescale"
    )
    aug = [layers.RandomFlip("horizontal_and_vertical"), layers.RandomRotation(0.2)]
    if variant == "improved":
        aug += [layers.RandomZoom(0.1), layers.RandomContrast(0.1)]
    data_augmentation = tf.keras.Sequential(aug, name="data_augmentation")

    body = [layers.Conv2D(32, (3, 3), activation="relu"), layers.MaxPooling2D((2, 2))]
    for _ in range(5):
        body += [layers.Conv2D(64, (3, 3), activation="relu"), layers.MaxPooling2D((2, 2))]
    head = [layers.Flatten()]
    if variant == "improved":
        head.append(layers.Dropout(0.4))
    head += [layers.Dense(64, activation="relu")]
    if variant == "improved":
        head.append(layers.Dropout(0.3))
    head.append(layers.Dense(len(CLASS_NAMES), activation="softmax"))

    model = models.Sequential([resize_and_rescale, data_augmentation, *body, *head], name=f"caulicare_{variant}")
    model.build(input_shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "improved"], required=True)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    data, split = load_split_arrays()
    (x_tr, y_tr), (x_va, y_va) = data["train"], data["val"]  # test set is never touched here

    model = build_model(args.variant)
    model.summary()
    params = model.count_params()
    print(f"input {model.input_shape} -> output {model.output_shape}; params {params:,}")

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["accuracy"])

    MODELS.mkdir(exist_ok=True)
    callbacks, class_weight = [], None
    if args.variant == "baseline":
        epochs = args.epochs or 100  # notebook: 100 epochs, no callbacks, final weights saved
        out_path = MODELS / "model_original_baseline.h5"
    else:
        epochs = args.epochs or 150
        out_path = MODELS / "model_improved.h5"
        counts = np.bincount(y_tr, minlength=len(CLASS_NAMES))
        class_weight = {i: len(y_tr) / (len(CLASS_NAMES) * c) for i, c in enumerate(counts)}
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5),
            tf.keras.callbacks.ModelCheckpoint(str(MODELS / "model_improved_best.h5"), monitor="val_loss", save_best_only=True),
        ]

    t0 = time.time()
    hist = model.fit(
        tf.data.Dataset.from_tensor_slices((x_tr, y_tr)).shuffle(len(x_tr), seed=SEED).batch(BATCH_SIZE).prefetch(2),
        validation_data=tf.data.Dataset.from_tensor_slices((x_va, y_va)).batch(BATCH_SIZE),
        epochs=epochs, class_weight=class_weight, callbacks=callbacks, verbose=2,
    )
    minutes = (time.time() - t0) / 60
    model.save(out_path)
    best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
    info = {
        "variant": args.variant, "artifact": out_path.name, "params": params, "epochs_run": len(hist.history["loss"]),
        "best_val_loss_epoch": best_epoch, "batch_size": BATCH_SIZE, "optimizer": "Adam(lr=1e-3)",
        "loss": "sparse_categorical_crossentropy", "class_weight": class_weight, "train_minutes": round(minutes, 1),
        "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()},
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"train_{args.variant}.json").write_text(json.dumps(info, indent=1))
    print(f"saved {out_path} after {info['epochs_run']} epochs in {minutes:.1f} min")


if __name__ == "__main__":
    main()
