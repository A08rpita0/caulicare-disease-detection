"""Inspect the raw dataset: counts, formats, sizes, corruption, exact + near duplicates."""
import collections
import json

from PIL import Image

from common import CLASS_NAMES, DATASET_DIR, REPORTS, ROOT, dhash, list_images, md5

NEAR_DUP_BITS = 4  # Hamming distance on 64-bit dHash


def main():
    REPORTS.mkdir(exist_ok=True)
    items = list_images()
    per_class = collections.Counter(CLASS_NAMES[c] for _, c in items)
    formats, sizes, modes, corrupt = collections.Counter(), collections.Counter(), collections.Counter(), []
    by_md5, hashes = collections.defaultdict(list), []
    for path, cls in items:
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im.load()
                formats[im.format] += 1
                sizes[im.size] += 1
                modes[im.mode] += 1
                hashes.append((path, cls, dhash(im)))
        except Exception as exc:  # noqa: BLE001
            corrupt.append(f"{path.relative_to(DATASET_DIR)}: {exc}")
            continue
        by_md5[md5(path)].append(path)

    exact = [[str(p.relative_to(DATASET_DIR)) for p in g] for g in by_md5.values() if len(g) > 1]
    near = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            d = bin(hashes[i][2] ^ hashes[j][2]).count("1")
            if d <= NEAR_DUP_BITS:
                near.append({"a": str(hashes[i][0].relative_to(DATASET_DIR)), "b": str(hashes[j][0].relative_to(DATASET_DIR)),
                             "distance": d, "cross_class": hashes[i][1] != hashes[j][1]})

    report = {
        "dataset_dir": DATASET_DIR.relative_to(ROOT).as_posix(),
        "total_images": len(items),
        "class_distribution": {c: per_class[c] for c in CLASS_NAMES},
        "formats": dict(formats),
        "color_modes": dict(modes),
        "image_dimensions": {f"{w}x{h}": n for (w, h), n in sizes.most_common()},
        "corrupted_images": corrupt,
        "exact_duplicate_groups": exact,
        "near_duplicate_pairs": near,
    }
    (REPORTS / "dataset_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "near_duplicate_pairs"}, indent=2))
    print(f"near-duplicate pairs (dHash <= {NEAR_DUP_BITS}): {len(near)}, cross-class: {sum(n['cross_class'] for n in near)}")
    for n in near[:15]:
        print("  ", n)


if __name__ == "__main__":
    main()
