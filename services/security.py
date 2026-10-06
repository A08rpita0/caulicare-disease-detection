"""Small security helpers: in-memory rate limiting and response headers."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from functools import wraps

from flask import current_app, jsonify, request

_UNITS = {"second": 1, "minute": 60, "hour": 3600}


class RateLimiter:
    """Sliding-window limiter keyed by client IP and endpoint.

    In-memory, so limits are per process. Good enough for a single-instance
    deployment; use Redis (e.g. Flask-Limiter) if you scale horizontally.
    """

    def __init__(self):
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    @staticmethod
    def parse(rule: str) -> tuple[int, int]:
        count, unit = rule.split("/")
        return int(count), _UNITS[unit.strip().rstrip("s")]

    def hit(self, key: str, rule: str) -> bool:
        limit, window = self.parse(rule)
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True


limiter = RateLimiter()


def rate_limited(config_key: str):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # Client address: a trusted edge-proxy header when configured (e.g.
            # CF-Connecting-IP on Render), otherwise remote_addr, which ProxyFix
            # (see app.py) derives from proxy-appended X-Forwarded-For entries.
            # Client-supplied X-Forwarded-For values are never trusted directly.
            header = current_app.config.get("CLIENT_IP_HEADER")
            ip = (request.headers.get(header, "").strip() if header else "") or request.remote_addr or "?"
            if not limiter.hit(f"{current_app.extensions['rate_ns']}:{fn.__name__}:{ip}", current_app.config[config_key]):
                return jsonify(success=False, error="Too many requests. Please wait a moment and try again."), 429
            return fn(*args, **kwargs)

        return wrapper

    return deco


CSP = (
    "default-src 'self'; "
    "img-src 'self' data: blob:; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "  # inline style attrs only; scripts stay strict
    "font-src 'self' https://fonts.gstatic.com; "
    "script-src 'self'; "
    "connect-src 'self'; "
    "frame-ancestors 'self'; "
    "base-uri 'self'; form-action 'self'"
)


def apply_security_headers(resp):
    resp.headers.setdefault("Content-Security-Policy", CSP)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    resp.headers.setdefault("Permissions-Policy", "camera=(self), microphone=(), geolocation=()")
    if request.is_secure:
        resp.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    origins = current_app.config.get("CORS_ORIGINS") or []
    origin = request.headers.get("Origin")
    if origin and origin in origins and request.path.startswith("/api/"):
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp
