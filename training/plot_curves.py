"""Plot training/validation loss and accuracy for each trained variant."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import REPORTS  # noqa: E402

for variant in ("baseline", "improved"):
    f = REPORTS / f"train_{variant}.json"
    if not f.exists():
        continue
    h = json.loads(f.read_text())["history"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["loss"], label="train"); ax[0].plot(h["val_loss"], label="validation"); ax[0].set_title(f"{variant}: loss")
    ax[1].plot(h["accuracy"], label="train"); ax[1].plot(h["val_accuracy"], label="validation"); ax[1].set_title(f"{variant}: accuracy")
    for a in ax:
        a.set_xlabel("epoch"); a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(REPORTS / f"curves_{variant}.png", dpi=110)
    print("wrote", REPORTS / f"curves_{variant}.png")
