"""Structured disease knowledge base with a small lexical retriever.

The curated records in data/diseases.json supply the disease information shown
next to a prediction, the disease library and the side-by-side comparison of
similar conditions. Each record is also split into small passages ("chunks"),
one per field item, and ranked with BM25 to power the library's symptom search
(e.g. "yellow V-shaped lesions"). At this size (~100 passages) BM25 is fast,
transparent and dependency-free.

All text comes from the knowledge file; nothing is generated.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

FIELD_TITLES = {
    "short_description": "Overview",
    "pathogen": "Cause / pathogen",
    "common_symptoms": "Symptoms",
    "visual_signs": "Visual signs",
    "possible_causes": "Causes and spread",
    "prevention": "Prevention",
    "general_management": "General management",
    "when_to_seek_expert_help": "When to seek expert help",
}
_STOP = set("a an the of to and or in on for is are be do does how what when why which can i my it this that with from at by as should could would will".split())
# Domain words that appear in almost every record (and in the "No disease" record's
# wording) and would otherwise dominate matching for questions like
# "What disease was detected?".
_GENERIC = {_w for _w in "disease diseases detected detect plant plants leaf leaves cauliflower model crop crops".split()}


# Everyday question words -> vocabulary used in the knowledge base (query side only).
_QUERY_SYNONYMS = {
    "nearby": ("spread",), "neighbour": ("spread",), "neighbor": ("spread",), "other": ("spread",),
    "affect": ("spread",), "infect": ("spread", "infection"), "contagiou": ("spread",), "transmit": ("spread",),
    "stop": ("prevent", "prevention"), "avoid": ("prevent", "prevention"), "cure": ("management",),
    "treat": ("management",), "treatment": ("management",), "sign": ("symptom",),
}


def _stem(t: str) -> str:
    """Very light suffix stripping so 'spreading'/'spreads'/'spread' match."""
    for suf in ("ing", "ed", "es", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[: -len(suf)]
    return t


def _tokens(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z]+", text.lower()) if t not in _STOP and t not in _GENERIC and len(t) > 1]


@dataclass
class Chunk:
    disease_id: str
    disease_name: str
    section: str
    text: str
    tokens: list[str]


class KnowledgeBase:
    def __init__(self, path: Path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.meta = raw["_meta"]
        self.sources = raw["sources"]
        self.diseases = raw["diseases"]
        self._by_id = {d["id"]: d for d in self.diseases}
        self._by_label = {d["label"].lower(): d for d in self.diseases}
        self.chunks = self._build_chunks()
        self._df = Counter(t for c in self.chunks for t in set(c.tokens))
        self._avgdl = sum(len(c.tokens) for c in self.chunks) / max(len(self.chunks), 1)

    # ---------------------------------------------------------------- lookups
    def get(self, key: str) -> dict | None:
        """Find a record by id ('black-rot') or model label ('Black Rot')."""
        if not key:
            return None
        k = key.strip().lower()
        return self._by_id.get(k) or self._by_label.get(k) or self._by_id.get(k.replace(" ", "-"))

    def public_record(self, d: dict) -> dict:
        rec = {k: v for k, v in d.items() if k != "source_keys"}
        rec["sources"] = [self.sources[s] for s in d.get("source_keys", []) if s in self.sources]
        return rec

    # -------------------------------------------------------------- retrieval
    def _build_chunks(self) -> list[Chunk]:
        chunks = []
        for d in self.diseases:
            for field, title in FIELD_TITLES.items():
                value = d.get(field)
                items = value if isinstance(value, list) else [value] if value else []
                for item in items:
                    text = f"{d['name']} - {title}: {item}"
                    chunks.append(Chunk(d["id"], d["name"], title, item, _tokens(text)))
        return chunks

    def retrieve(self, query: str, disease_id: str | None = None, k: int = 8) -> list[Chunk]:
        q = _tokens(query)
        q += [_stem(syn) for t in q for syn in _QUERY_SYNONYMS.get(t, ())]
        focus = {disease_id} if disease_id else set()
        lowered = query.lower()
        for d in self.diseases:  # diseases named in the question (e.g. comparisons) get the same focus
            if d["id"] != "no-disease" and (d["name"].lower() in lowered or d["label"].lower() in lowered):
                focus.add(d["id"])
        # With a detected or named disease, retrieve only from those records so other
        # classes' wording cannot leak in; otherwise search the whole knowledge base.
        pool = [c for c in self.chunks if c.disease_id in focus] if focus else self.chunks
        n = len(self.chunks)
        k1, b = 1.4, 0.75
        scored = []
        for c in pool:
            tf = Counter(c.tokens)
            s = 0.0
            for t in q:
                if t in tf:
                    idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                    s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(c.tokens) / self._avgdl))
            if focus:
                s += 0.01  # vague questions still return the focused record
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        ranked = [c for _, c in scored]
        if len(focus) > 1:  # comparison: interleave so every named disease is represented
            groups = [[c for c in ranked if c.disease_id == fid] for fid in sorted(focus)]
            ranked = [c for tier in zip(*groups) for c in tier]
        # Always ground on each focused disease's overview.
        overviews = [c for c in self.chunks if c.disease_id in focus and c.section == "Overview"]
        ranked = overviews + [c for c in ranked if c not in overviews]
        return ranked[:k]

    def search(self, query: str, k: int = 8) -> list[dict]:
        """Rank knowledge passages for a free-text symptom or topic query."""
        if not _tokens(query):
            return []
        return [
            {"disease_id": c.disease_id, "disease": c.disease_name, "section": c.section, "text": c.text}
            for c in self.retrieve(query, k=k)
        ]

    def compare(self, first: str, second: str) -> dict | None:
        """Side-by-side visual information for two conditions, straight from the records."""
        a, b = self.get(first), self.get(second)
        if not a or not b:
            return None
        keys = ("short_description", "visual_signs", "common_symptoms")
        return {"conditions": [{"id": d["id"], "name": d["name"], **{k: d.get(k) for k in keys}} for d in (a, b)]}
