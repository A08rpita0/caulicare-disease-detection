"""Request IDs, per-stage timing, structured request logs and in-memory service statistics.

Nothing here stores images or user identifiers: a log line carries the request id,
route, status, latency and the outcome of the analysis (decision, predicted class,
confidence level), never a file name, address or pixel data.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections import Counter, deque
from contextlib import contextmanager

from flask import Flask, g, request

log = logging.getLogger("caulicare.request")

# A caller may send its own id (to correlate logs across services); anything else is replaced.
_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
# Polled by the platform / browser; counting them would drown real traffic.
_UNTRACKED = {"/api/health", "/api/stats"}


def current_request_id() -> str | None:
    return getattr(g, "request_id", None)


def annotate(**fields) -> None:
    """Attach outcome fields (decision, disease, ...) to this request's log line and statistics."""
    if not hasattr(g, "outcome"):
        g.outcome = {}
    g.outcome.update({k: v for k, v in fields.items() if v is not None})


class Stopwatch:
    """Wall-clock time per named stage, in milliseconds."""

    def __init__(self):
        self._t0 = time.perf_counter()
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - t) * 1000)

    def add(self, name: str, ms: float) -> None:
        self.stages[name] = round(self.stages.get(name, 0.0) + ms, 1)

    def merge(self, stages: dict | None, prefix: str = "") -> None:
        for k, v in (stages or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                self.add(prefix + k, float(v))

    def total_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 1)

    def as_dict(self) -> dict:
        return {**self.stages, "total": self.total_ms()}


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[idx]


class Metrics:
    """Thread-safe request counters and recent-latency percentiles (per process)."""

    WINDOW = 500

    def __init__(self):
        self._lock = threading.Lock()
        self.started = time.time()
        self.requests = 0
        self.by_status: Counter = Counter()
        self.decisions: Counter = Counter()
        self.confidence_levels: Counter = Counter()
        self.predictions: Counter = Counter()
        self.cached = 0
        self._latency: dict[str, deque] = {}

    def record(self, endpoint: str, status: int, latency_ms: float, outcome: dict | None = None) -> None:
        outcome = outcome or {}
        with self._lock:
            self.requests += 1
            self.by_status[str(status)] += 1
            self._latency.setdefault(endpoint, deque(maxlen=self.WINDOW)).append(latency_ms)
            if outcome.get("decision"):
                self.decisions[outcome["decision"]] += 1
            if outcome.get("confidence_level"):
                self.confidence_levels[outcome["confidence_level"]] += 1
            if outcome.get("disease"):
                self.predictions[outcome["disease"]] += 1
            if outcome.get("cached"):
                self.cached += 1

    def snapshot(self) -> dict:
        with self._lock:
            endpoints = {}
            for name, window in self._latency.items():
                values = sorted(window)
                endpoints[name] = {
                    "count": len(values),
                    "latency_ms": {"p50": _percentile(values, 0.50), "p95": _percentile(values, 0.95),
                                   "max": values[-1] if values else 0.0},
                }
            return {
                "uptime_seconds": round(time.time() - self.started),
                "requests_total": self.requests,
                "by_status": dict(self.by_status),
                "decisions": dict(self.decisions),
                "confidence_levels": dict(self.confidence_levels),
                "predicted_classes": dict(self.predictions),
                "served_from_cache": self.cached,
                "endpoints": endpoints,
                "latency_window": f"last {self.WINDOW} requests per endpoint",
            }


def init_telemetry(app: Flask) -> Metrics:
    metrics = Metrics()
    app.extensions["metrics"] = metrics

    @app.before_request
    def _start():
        supplied = request.headers.get("X-Request-ID", "")
        g.request_id = supplied if _ID_RE.match(supplied) else uuid.uuid4().hex[:16]
        g.started = time.perf_counter()
        g.outcome = {}

    @app.after_request
    def _finish(resp):
        rid = getattr(g, "request_id", None)
        if rid:
            resp.headers["X-Request-ID"] = rid
        path = request.path
        if path.startswith("/api/") and path not in _UNTRACKED and hasattr(g, "started"):
            latency = round((time.perf_counter() - g.started) * 1000, 1)
            endpoint = f"{request.method} {path}" if not path.startswith("/api/diseases/") else f"{request.method} /api/diseases/<name>"
            metrics.record(endpoint, resp.status_code, latency, g.outcome)
            log.info(json.dumps({"event": "request", "request_id": rid, "method": request.method, "path": path,
                                 "status": resp.status_code, "latency_ms": latency, **g.outcome},
                                separators=(",", ":")))
        return resp

    return metrics
