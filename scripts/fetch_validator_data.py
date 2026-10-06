"""Download the public image sets used to train and test the cauliflower-leaf validator.

Nothing downloaded here is committed. Images are resized (longest side 640 px)
and written to <out>/images/<source>/, with one manifest.csv row per image.

  positives  VegNet (local copy, see training/README.md)                 CC BY 4.0
             Durjoy et al. 2025, cauliflower leaf diseases               CC BY 4.0
  negatives  PlantDoc (13 other crops, field photos)                     CC BY 4.0
             iBean leaves (Makerere AI Lab)                              MIT
             Oxford-IIIT Pet (cats and dogs)                             CC BY-SA 4.0
             COCO 2017 val (people, vehicles, buildings, food, objects)  Flickr licences, research use
             Oxford Flowers 102                                          research use, no explicit licence
             Synthetic documents and screenshots (generated here)        -

    python scripts/fetch_validator_data.py --out C:/caulicare-local-assets/validator_data \
        --vegnet "path/to/Original Dataset"
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import textwrap
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
MAX_SIDE = 640

# (source name, label, repo, parquet files, max images per file, licence)
SOURCES = [
    ("durjoy_cauliflower", 1, "Project-AgML/cauliflower_leaf_disease_classification",
     ["data/train-00000-of-00011.parquet", "data/train-00005-of-00011.parquet",
      "data/train-00009-of-00011.parquet"], 400, "CC BY 4.0"),
    ("plantdoc", 0, "agyaatcoder/PlantDoc",
     ["data/test-00000-of-00001.parquet", "data/train-00000-of-00002.parquet"], 700, "CC BY 4.0"),
    ("beans", 0, "AI-Lab-Makerere/beans",
     ["data/validation-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"], 200, "MIT"),
    ("pets", 0, "timm/oxford-iiit-pet", ["data/test-00000-of-00001.parquet"], 600, "CC BY-SA 4.0"),
    ("coco", 0, "detection-datasets/coco", ["data/val-00000-of-00002-c4f2e391ee4aba11.parquet"], 900,
     "Flickr licences (research use)"),
    ("flowers", 0, "dpdl-benchmark/oxford_flowers102", ["data/validation-00000-of-00001.parquet"], 400,
     "research use"),
]


def download(repo: str, path: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(".part")
    for attempt in range(8):  # resume after dropped connections
        done = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            with requests.get(HF.format(repo=repo, path=path), stream=True, timeout=120, headers=headers) as r:
                if r.status_code == 416:  # already complete
                    break
                r.raise_for_status()
                with open(tmp, "ab" if r.status_code == 206 else "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            break
        except requests.RequestException as exc:
            print(f"  retry {attempt + 1} for {path}: {exc.__class__.__name__}", flush=True)
    else:
        raise RuntimeError(f"download failed: {repo}/{path}")
    tmp.replace(dest)
    return dest


def save(img: Image.Image, path: Path) -> tuple[int, int]:
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=90)
    return img.size


def image_column(schema) -> str:
    for name in ("image", "img"):
        if name in schema.names:
            return name
    raise KeyError(f"no image column in {schema.names}")


def from_parquet(name, label, repo, files, per_file, licence, out: Path, rows: list, rng: random.Random):
    import pyarrow.parquet as pq

    for fi, f in enumerate(files):
        local = download(repo, f, out / "raw" / repo.replace("/", "__") / Path(f).name)
        pf = pq.ParquetFile(local)
        n = pf.metadata.num_rows
        pick = set(rng.sample(range(n), min(per_file, n)))
        offset = 0
        for batch in pf.iter_batches(batch_size=32):
            col = image_column(batch.schema)
            other = {c: batch.column(c).to_pylist() for c in batch.schema.names if c in ("label", "objects")}
            for j, cell in enumerate(batch.column(col).to_pylist()):
                i = offset + j
                if i not in pick:
                    continue
                try:
                    img = Image.open(io.BytesIO(cell["bytes"]))
                    img.load()
                except Exception:  # noqa: BLE001 - skip undecodable rows
                    continue
                rel = Path("images") / name / f"{name}-{fi}-{i:05d}.jpg"
                w, h = save(img, out / rel)
                sub = str(other.get("label", [""] * (j + 1))[j]) if "label" in other else ""
                rows.append({"path": rel.as_posix(), "source": name, "label": label, "subclass": sub,
                             "licence": licence, "width": w, "height": h})
            offset += batch.num_rows
        print(f"{name}: {sum(r['source'] == name for r in rows)} images so far")


def from_vegnet(vegnet: Path, out: Path, rows: list):
    import json

    splits = json.loads((Path(__file__).resolve().parents[1] / "data" / "splits.json").read_text())
    # splits.json maps split -> list of relative paths (class/file)
    for split, items in splits["splits"].items():
        for item in items:
            rel_src = item["path"]
            src = vegnet / rel_src
            rel = Path("images") / "vegnet" / split / rel_src
            w, h = save(Image.open(src), out / rel)
            rows.append({"path": rel.as_posix(), "source": "vegnet", "label": 1, "subclass": Path(rel_src).parent.name,
                         "licence": "CC BY 4.0", "width": w, "height": h, "split": split})
    print(f"vegnet: {sum(r['source'] == 'vegnet' for r in rows)} images")


WORDS = ("report invoice total amount date customer order account payment balance summary quarterly revenue "
         "analysis table figure section introduction method results discussion conclusion reference page "
         "login password settings profile dashboard notifications search home menu help contact").split()


def synthetic(out: Path, rows: list, rng: random.Random, n: int = 160):
    """Documents, slides and app screenshots rendered from scratch (no third-party content)."""
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        big = ImageFont.truetype("arial.ttf", 34)
    except OSError:
        font = big = ImageFont.load_default()
    for i in range(n):
        kind = ("document", "screenshot", "slide", "spreadsheet")[i % 4]
        w, h = rng.choice([(800, 1100), (1280, 800), (1024, 768), (720, 1280)])
        bg = (255, 255, 255) if kind != "slide" else tuple(rng.randint(0, 255) for _ in range(3))
        img = Image.new("RGB", (w, h), bg)
        d = ImageDraw.Draw(img)
        fg = (20, 20, 20) if sum(bg) > 380 else (240, 240, 240)
        y = 40
        if kind == "screenshot":
            accent = tuple(rng.randint(0, 200) for _ in range(3))
            d.rectangle([0, 0, w, 70], fill=accent)
            d.text((30, 20), " ".join(rng.choices(WORDS, k=3)).title(), font=big, fill=(255, 255, 255))
            y = 110
            for b in range(rng.randint(2, 5)):
                x0 = 30 + b * (w // 5)
                d.rounded_rectangle([x0, y, x0 + w // 6, y + 120], 12, outline=accent, width=3)
            y += 160
        if kind == "slide":
            d.text((60, 60), " ".join(rng.choices(WORDS, k=4)).title(), font=big, fill=fg)
            y = 160
        if kind == "spreadsheet":
            for r_ in range(0, h, 32):
                d.line([0, r_, w, r_], fill=(200, 200, 200))
            for c in range(0, w, 120):
                d.line([c, 0, c, h], fill=(200, 200, 200))
            for r_ in range(1, h // 32):
                for c in range(0, w // 120):
                    d.text((c * 120 + 6, r_ * 32 + 6), str(rng.randint(0, 99999)), font=font, fill=fg)
        else:
            while y < h - 60:
                line = " ".join(rng.choices(WORDS, k=rng.randint(6, 14)))
                for part in textwrap.wrap(line, 60):
                    d.text((50, y), part.capitalize(), font=font, fill=fg)
                    y += 28
                y += 12
        rel = Path("images") / "synthetic_docs" / f"{kind}-{i:04d}.jpg"
        sw, sh = save(img, out / rel)
        rows.append({"path": rel.as_posix(), "source": "synthetic_docs", "label": 0, "subclass": kind,
                     "licence": "generated", "width": sw, "height": sh})
    print(f"synthetic_docs: {n} images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--vegnet", required=True, type=Path)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    rng = random.Random(42)
    rows: list[dict] = []
    if not args.only or "vegnet" in args.only:
        from_vegnet(args.vegnet, args.out, rows)
    for spec in SOURCES:
        if not args.only or spec[0] in args.only:
            from_parquet(*spec, args.out, rows, rng)
    if not args.only or "synthetic_docs" in args.only:
        synthetic(args.out, rows, rng)
    keys = ["path", "source", "label", "subclass", "licence", "width", "height", "split"]
    with open(args.out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
