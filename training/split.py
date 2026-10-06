"""Deterministic, duplicate-aware, class-stratified train/val/test split.

Images whose 64-bit dHash differ by <= GROUP_BITS (and exact MD5 duplicates)
within the same class are grouped (union-find), and whole groups are assigned
to a single split, so burst shots of the same leaf cannot leak across splits.
"""
import json
import random
from collections import defaultdict

from PIL import Image

from common import CLASS_NAMES, DATASET_DIR, SEED, SPLIT_FILE, dhash, list_images, md5

GROUP_BITS = 6
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def main():
    items = list_images()
    hashes, md5s = [], []
    for p, _ in items:
        with Image.open(p) as im:
            hashes.append(dhash(im))
        md5s.append(md5(p))

    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i][1] != items[j][1]:
                continue
            if md5s[i] == md5s[j] or bin(hashes[i] ^ hashes[j]).count("1") <= GROUP_BITS:
                parent[find(i)] = find(j)

    groups_by_class = defaultdict(dict)
    for i, (_, c) in enumerate(items):
        groups_by_class[c].setdefault(find(i), []).append(i)

    rng = random.Random(SEED)
    assignment = {}
    for c in range(len(CLASS_NAMES)):
        groups = sorted(groups_by_class[c].values(), key=lambda g: min(g))
        rng.shuffle(groups)
        n = sum(len(g) for g in groups)
        targets = {"test": round(n * FRACTIONS["test"]), "val": round(n * FRACTIONS["val"])}
        filled = {"test": 0, "val": 0}
        for g in groups:
            split = next((s for s in ("test", "val") if filled[s] + len(g) <= targets[s]), "train")
            if split != "train":
                filled[split] += len(g)
            for i in g:
                assignment[i] = split

    splits = {s: [] for s in FRACTIONS}
    for i, (p, c) in enumerate(items):
        splits[assignment[i]].append({"path": p.relative_to(DATASET_DIR).as_posix(), "label": c})

    # Leakage audit across splits.
    idx_split = {i: assignment[i] for i in range(len(items))}
    leaks = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if idx_split[i] != idx_split[j] and (md5s[i] == md5s[j] or bin(hashes[i] ^ hashes[j]).count("1") <= GROUP_BITS):
                leaks += items[i][1] == items[j][1]
    counts = {s: {CLASS_NAMES[c]: sum(1 for e in v if e["label"] == c) for c in range(4)} | {"total": len(v)} for s, v in splits.items()}
    out = {"seed": SEED, "group_bits": GROUP_BITS, "class_names": CLASS_NAMES, "counts": counts,
           "same_class_near_duplicates_across_splits": leaks, "splits": splits}
    SPLIT_FILE.write_text(json.dumps(out, indent=1))
    print(json.dumps(counts, indent=1))
    print("same-class duplicate pairs crossing splits:", leaks)


if __name__ == "__main__":
    main()
