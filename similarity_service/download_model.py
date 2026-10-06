"""Download the DINOv2-small ONNX encoder used for visual similarity search.

Source: https://huggingface.co/onnx-community/dinov2-small (ONNX export of
facebook/dinov2-small, Apache-2.0). The file is verified against a pinned
SHA-256 before it is used.

    python similarity_service/download_model.py

Run once at build time (Render build command / Dockerfile), never at startup.
A checksum mismatch exits with status 1, which fails the build, so an
unexpected model is never deployed.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parent / "models"
BASE = "https://huggingface.co/onnx-community/dinov2-small/resolve/main"
FILES = {
    "dinov2-small.onnx": ("onnx/model.onnx", "f22797eabf810a75e41de68d378541ebea372122b25c4ce3ef25ff618250c20a"),
    "preprocessor_config.json": ("preprocessor_config.json", "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, (remote, digest) in FILES.items():
        target = DEST / name
        if target.exists() and sha256(target) == digest:
            print(f"ok      {name} (already present)")
            continue
        tmp = target.with_suffix(target.suffix + ".part")
        print(f"fetch   {name} ...")
        with urllib.request.urlopen(f"{BASE}/{remote}", timeout=300) as resp, open(tmp, "wb") as fh:  # noqa: S310
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        if sha256(tmp) != digest:
            tmp.unlink()
            print(f"error   {name}: checksum mismatch, file discarded", file=sys.stderr)
            return 1
        tmp.replace(target)
        print(f"ok      {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
