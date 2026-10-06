"""Client for the CauliCare Similarity service (DINOv2 + FAISS, see similarity_service/).

The main app never loads the embedding model, the reference index or the
cauliflower-leaf validator itself: it sends the uploaded image to the service.
This client never raises.

  verify()  the cauliflower-leaf check that gates the disease model. Any failure
            (not configured, asleep, timeout, malformed answer) returns
            {"checked": False}; the caller must then refuse to analyse (fail closed).
  search()  similar reference images, an extra section of the result. A failure
            becomes {"available": False, "reason": ...} and the analysis continues.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
import uuid

log = logging.getLogger(__name__)

_ID = re.compile(r"^[a-z0-9-]{1,64}$")
_THUMB_PREFIX = "data:image/jpeg;base64,"
_MAX_THUMB_CHARS = 300_000
NOT_CONFIGURED = "Visual similarity search is not set up on this server."
UNAVAILABLE = "Visual similarity search is temporarily unavailable. Your disease result is complete."


class SimilarityClient:
    def __init__(self, base_url: str = "", timeout: float = 30.0, token: str = "", k: int = 5,
                 labels: list[str] | None = None, verify_timeout: float = 75.0):
        self.base_url = (base_url or "").strip().rstrip("/")
        self.timeout = timeout
        # Longer than the search timeout: a free-tier service may be waking from sleep,
        # and without an answer the upload cannot be analysed at all.
        self.verify_timeout = verify_timeout
        self.token = token
        self.k = k
        self.labels = set(labels or [])
        self.last_status: dict = {"reachable": None, "checked": None}
        self._warm_lock = threading.Lock()
        self._last_warm = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    # ------------------------------------------------------------------ verify
    def verify(self, data: bytes, filename: str = "image", mimetype: str = "application/octet-stream") -> dict:
        """{"checked": True, "is_cauliflower_leaf": bool, "score": float, "threshold": float}
        or {"checked": False} when the check could not be made."""
        if not self.configured:
            return {"checked": False}
        try:
            raw = self._post_image("/verify", data, filename, mimetype, self.verify_timeout, k=None)
            body = json.loads(raw)
            ok, score, thr = body.get("is_cauliflower_leaf"), body.get("score"), body.get("threshold")
            if body.get("success") is not True or not isinstance(ok, bool):
                raise ValueError("malformed response")
            for v in (score, thr):
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= float(v) <= 1.0:
                    raise ValueError("bad score")
            if ok != (float(score) >= float(thr)):  # the verdict must agree with its own numbers
                raise ValueError("inconsistent verdict")
        except Exception as exc:  # noqa: BLE001 - reported as "not checked"; the caller fails closed
            log.warning("Leaf check unavailable: %s%s", exc.__class__.__name__,
                        f" {exc.code}" if isinstance(exc, urllib.error.HTTPError) else "")
            self._record(False)
            return {"checked": False}
        self._record(True)
        return {"checked": True, "is_cauliflower_leaf": ok, "score": round(float(score), 4),
                "threshold": round(float(thr), 4)}

    # ------------------------------------------------------------------ search
    def search(self, data: bytes, filename: str = "image", mimetype: str = "application/octet-stream") -> dict:
        if not self.configured:
            return {"available": False, "reason": NOT_CONFIGURED}
        t0 = time.perf_counter()
        try:
            raw = self._post_image("/search", data, filename, mimetype, self.timeout, k=self.k)
            out = self._validate(json.loads(raw))
        except Exception as exc:  # noqa: BLE001 - any failure only disables this section
            log.warning("Similarity service unavailable: %s%s", exc.__class__.__name__,
                        f" {exc.code}" if isinstance(exc, urllib.error.HTTPError) else "")
            self._record(False)
            return {"available": False, "reason": UNAVAILABLE}
        self._record(True)
        out["timing_ms"]["service_round_trip"] = round((time.perf_counter() - t0) * 1000, 1)
        return {"available": True, **out}

    def _post_image(self, path: str, data: bytes, filename: str, mimetype: str, timeout: float,
                    k: int | None) -> bytes:
        boundary = uuid.uuid4().hex
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", filename or "image")[:80]
        safe_type = mimetype if re.fullmatch(r"image/[a-z0-9.+-]+", mimetype or "") else "application/octet-stream"
        k_field = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"k\"\r\n\r\n{k}\r\n".encode()
                   if k is not None else b"")
        body = b"".join([
            k_field,
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{safe_name}\"\r\n"
            f"Content-Type: {safe_type}\r\n\r\n".encode(),
            data,
            f"\r\n--{boundary}--\r\n".encode(),
        ])
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        if self.token:
            headers["X-Service-Token"] = self.token
        req = urllib.request.Request(f"{self.base_url}{path}", data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - URL from server config
            if resp.status != 200:
                raise ValueError(f"status {resp.status}")
            return resp.read(5_000_000)

    def _validate(self, body) -> dict:
        """Accept only the documented shape; drop anything else."""
        if not isinstance(body, dict) or body.get("success") is not True or not isinstance(body.get("results"), list):
            raise ValueError("malformed response")
        if len(body["results"]) > self.k:
            raise ValueError("too many results")
        results = []
        for r in body["results"]:
            if not isinstance(r, dict):
                raise ValueError("malformed result")
            rid, label, sim, thumb = r.get("id"), r.get("label"), r.get("similarity"), r.get("thumbnail")
            if not (isinstance(rid, str) and _ID.match(rid)):
                raise ValueError("bad id")
            if not isinstance(label, str) or (self.labels and label not in self.labels):
                raise ValueError("bad label")
            if isinstance(sim, bool) or not isinstance(sim, (int, float)) or not 0.0 <= float(sim) <= 1.0:
                raise ValueError("bad similarity")
            if not (isinstance(thumb, str) and thumb.startswith(_THUMB_PREFIX) and len(thumb) <= _MAX_THUMB_CHARS
                    and re.fullmatch(r"[A-Za-z0-9+/=]+", thumb[len(_THUMB_PREFIX):])):
                raise ValueError("bad thumbnail")
            results.append({"id": rid, "label": label, "similarity": round(float(sim), 4), "thumbnail": thumb})
        timing = body.get("timing_ms") if isinstance(body.get("timing_ms"), dict) else {}
        out = {
            "results": results,
            "timing_ms": {k: float(v) for k, v in timing.items()
                          if k in ("embedding", "search") and isinstance(v, (int, float)) and not isinstance(v, bool)},
        }
        for key in ("model", "backend"):
            if isinstance(body.get(key), str):
                out[key] = body[key][:80]
        if isinstance(body.get("reference_images"), int):
            out["reference_images"] = body["reference_images"]
        return out

    # ------------------------------------------------------------- warm-up
    def warm_up(self, min_interval: float = 120.0) -> None:
        """Ping /health in the background so a sleeping free-tier service starts waking
        up while the user is still choosing a photo. Throttled; never blocks."""
        if not self.configured:
            return
        with self._warm_lock:
            now = time.monotonic()
            if now - self._last_warm < min_interval:
                return
            self._last_warm = now
        threading.Thread(target=self._ping, daemon=True).start()

    def _ping(self) -> None:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=120) as resp:  # noqa: S310
                self._record(resp.status == 200)
        except Exception:  # noqa: BLE001
            self._record(False)

    def _record(self, ok: bool) -> None:
        self.last_status = {"reachable": ok, "checked": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
