"""Throw bad inputs at the API and record how it responds.

    python scripts/robustness_report.py            # prints a table, writes reports/robustness.json

Each case states the HTTP status and error `code` the service must answer with; the script exits 1
if any case differs, so it doubles as a regression check. The real CNN is used. The cauliflower-leaf
check is replaced by a stub (accept / reject / unavailable) because it is a call to the separate
similarity service; its own behaviour is covered by tests/test_similarity_service.py.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def main() -> int:
    import logging

    import numpy as np
    from PIL import Image, ImageFilter

    from app import create_app
    from benchmark_inference import synthetic_photo
    from config import TestConfig
    from services.predictor import Predictor
    from services.similarity_client import SimilarityClient

    logging.disable(logging.CRITICAL)

    class Stub(SimilarityClient):
        def __init__(self, mode: str):
            super().__init__("")
            self.mode = mode

        def verify(self, data, filename="image", mimetype="application/octet-stream"):
            if self.mode == "unavailable":
                return {"checked": False}
            ok = self.mode == "accept"
            return {"checked": True, "is_cauliflower_leaf": ok, "score": 0.95 if ok else 0.05, "threshold": 0.717}

    class Cfg(TestConfig):
        RATE_LIMIT_PREDICT = "1000000/minute"

    predictor = Predictor(ROOT / "models" / "model1.h5", onnx_path=ROOT / "models" / "model1.onnx")
    predictor.load()
    clients = {m: create_app(Cfg, predictor=predictor, similarity=Stub(m)).test_client()
               for m in ("accept", "reject", "unavailable")}

    def encode(img: Image.Image, fmt="JPEG") -> bytes:
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return buf.getvalue()

    leaf = synthetic_photo(7)
    leaf_img = Image.open(io.BytesIO(leaf)).convert("RGB")
    rng = np.random.default_rng(0)
    noise = Image.fromarray(rng.integers(0, 255, (2200, 1800, 3), dtype=np.uint8))
    grey = lambda v, size=(800, 800): Image.new("RGB", size, (v, v, v))  # noqa: E731
    bomb = Image.new("1", (12000, 8000))

    # (name, client, field, filename, mimetype, bytes, expected status, expected code or None)
    cases = [
        ("Valid JPEG photo", "accept", "image", "leaf.jpg", "image/jpeg", leaf, 200, None),
        ("Valid PNG photo", "accept", "image", "leaf.png", "image/png", encode(leaf_img, "PNG"), 200, None),
        ("Valid WEBP photo", "accept", "image", "leaf.webp", "image/webp", encode(leaf_img, "WEBP"), 200, None),
        ("Browser sends no content type", "accept", "image", "leaf.jpg", "application/octet-stream", leaf, 200, None),
        ("No file in the request", "accept", None, None, None, b"", 400, "no_file"),
        ("Wrong form field name", "accept", "photo", "leaf.jpg", "image/jpeg", leaf, 400, "no_file"),
        ("Empty file", "accept", "image", "leaf.jpg", "image/jpeg", b"", 400, "empty_file"),
        ("Text file renamed .jpg", "accept", "image", "leaf.jpg", "image/jpeg", b"this is not an image" * 50, 422, "corrupt_image"),
        ("Truncated JPEG (600 bytes)", "accept", "image", "leaf.jpg", "image/jpeg", leaf[:600], 422, "corrupt_image"),
        ("Half of a JPEG", "accept", "image", "leaf.jpg", "image/jpeg", leaf[: len(leaf) // 2], 422, "corrupt_image"),
        ("PDF document", "accept", "image", "scan.pdf", "application/pdf", b"%PDF-1.4\n" + b"0" * 500, 415, "unsupported_extension"),
        ("GIF image", "accept", "image", "leaf.gif", "image/gif", encode(Image.new("RGB", (300, 300)), "GIF"), 415, "unsupported_extension"),
        ("GIF renamed .png", "accept", "image", "leaf.png", "image/png", encode(Image.new("RGB", (300, 300)), "GIF"), 415, "unsupported_format"),
        ("PNG sent as application/pdf", "accept", "image", "leaf.png", "application/pdf", encode(leaf_img, "PNG"), 415, "unsupported_mimetype"),
        ("File over the size limit (> 8 MB)", "accept", "image", "big.png", "image/png", encode(noise, "PNG"), 413, "file_too_large"),
        ("Decompression bomb (96 megapixels)", "accept", "image", "bomb.png", "image/png", encode(bomb, "PNG"), 422, "too_many_pixels"),
        ("Tiny image (64 x 64)", "accept", "image", "tiny.jpg", "image/jpeg", encode(leaf_img.resize((64, 64))), 422, "low_quality"),
        ("Blank white frame", "accept", "image", "white.jpg", "image/jpeg", encode(grey(255)), 422, "low_quality"),
        ("Almost black frame", "accept", "image", "dark.jpg", "image/jpeg", encode(leaf_img.point(lambda v: v // 14)), 422, "low_quality"),
        ("Very blurry photo", "accept", "image", "blur.jpg", "image/jpeg", encode(leaf_img.filter(ImageFilter.GaussianBlur(40))), 422, "low_quality"),
        ("Not a cauliflower leaf (validator says no)", "reject", "image", "dog.jpg", "image/jpeg", leaf, 422, "not_cauliflower_leaf"),
        ("Leaf check cannot be made (fails closed)", "unavailable", "image", "leaf.jpg", "image/jpeg", leaf, 503, "leaf_check_unavailable"),
    ]

    rows, failures = [], 0
    for name, who, field, fname, mime, data, want_status, want_code in cases:
        form = {field: (io.BytesIO(data), fname, mime)} if field else {}
        r = clients[who].post("/api/predict", data=form, content_type="multipart/form-data")
        body = r.get_json(silent=True) or {}
        # Refusals for poor quality / non-leaf carry the reason in `code`; so do plain errors.
        code = body.get("code") if r.status_code != 200 else None
        ok = r.status_code == want_status and (want_code is None or code == want_code)
        failures += not ok
        text = r.get_data(as_text=True)
        rows.append({"case": name, "expected_status": want_status, "expected_code": want_code,
                     "status": r.status_code, "code": code, "decision": body.get("decision"),
                     "result": (body.get("result") or {}).get("label"), "passed": ok,
                     "leaks_internals": any(w in text for w in ("Traceback", "File \"", ".py\"", "C:\\"))})

    leaks = sum(r["leaks_internals"] for r in rows)
    width = max(len(r["case"]) for r in rows)
    print(f"{'case':<{width}}  {'expected':<28}{'actual':<28}decision")
    for r in rows:
        exp = f"{r['expected_status']} {r['expected_code'] or ''}".strip()
        got = f"{r['status']} {r['code'] or ''}".strip()
        flag = "" if r["passed"] else "   <-- MISMATCH"
        print(f"{r['case']:<{width}}  {exp:<28}{got:<28}{r['decision'] or ''}{flag}")
    print(f"\n{len(rows) - failures}/{len(rows)} cases answered as specified; {leaks} responses exposed internals.")

    out = ROOT / "reports" / "robustness.json"
    out.write_text(json.dumps({"cases": rows, "passed": len(rows) - failures, "total": len(rows),
                               "responses_exposing_internals": leaks}, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 1 if failures or leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
