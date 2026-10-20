"""Small in-memory LRU cache for finished analyses, keyed by the SHA-256 of the uploaded bytes.

The same photo analysed twice (a retry, a double click, a re-upload from History) returns
instantly instead of repeating decoding, the leaf check, the CNN and the similarity search.
Only the bytes' hash and the finished result are kept; the image itself is never stored.
"""
from __future__ import annotations

import copy
import hashlib
import threading
from collections import OrderedDict


def fingerprint(data: bytes, *parts: str) -> str:
    h = hashlib.sha256(data).hexdigest()
    return ":".join([h, *parts])


class ResultCache:
    def __init__(self, max_items: int = 0):
        self.max_items = max(0, int(max_items))
        self._items: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.max_items > 0

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        with self._lock:
            value = self._items.get(key)
            if value is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return copy.deepcopy(value)

    def put(self, key: str, value: dict) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._items[key] = copy.deepcopy(value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            lookups = self.hits + self.misses
            return {"enabled": self.enabled, "max_items": self.max_items, "items": len(self._items),
                    "hits": self.hits, "misses": self.misses,
                    "hit_rate": round(self.hits / lookups, 3) if lookups else None}
