#!/usr/bin/env python3
"""Download and materialize the small builtin datasets used by mode B.

The script uses only Python's standard library. It downloads official/public
dataset archives, extracts small teaching subsets, and writes them under:

  backend/spring/datasets/builtin/
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import pickle
import random
import shutil
import struct
import sys
import tarfile
import tempfile
import urllib.request
import zlib
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SPRING_ROOT = SCRIPT_DIR.parent
DATA_ROOT = SPRING_ROOT / "datasets" / "builtin"
ARCHIVE_ROOT = SPRING_ROOT / "datasets" / "_archives"

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}
CIFAR10_URLS = [
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
]
IRIS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"


def main() -> int:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)

    download_mnist()
    download_cifar10()
    download_iris()
    generate_points()

    write_catalog()
    print(f"Builtin datasets are stored in: {DATA_ROOT}")
    return 0


def download_mnist() -> None:
    target = DATA_ROOT / "mnist-1000"
    images_dir = target / "images"
    if count_files(images_dir, "*.png") >= 70000:
        print("MNIST full dataset already exists; skipping.")
        return

    reset_dir(images_dir)
    per_label = {str(i): 0 for i in range(10)}
    for split in ("train", "test"):
        image_archive = download(MNIST_URLS[f"{split}_images"], ARCHIVE_ROOT / f"mnist-{split}-images.gz")
        label_archive = download(MNIST_URLS[f"{split}_labels"], ARCHIVE_ROOT / f"mnist-{split}-labels.gz")
        exported = export_mnist_split(image_archive, label_archive, images_dir, split, per_label)
        print(f"MNIST {split} exported: {exported} images.")

    write_metadata(target, {
        "id": "mnist-1000",
        "sourceUrl": list(MNIST_URLS.values()),
        "sampleCount": sum(per_label.values()),
        "labels": list(per_label.keys()),
        "layout": "images/{label}/*.png",
    })
    print("MNIST full dataset ready.")


def download_cifar10() -> None:
    target = DATA_ROOT / "cifar10-500"
    images_dir = target / "images"
    if count_files(images_dir, "*.png") >= 60000:
        print("CIFAR-10 full dataset already exists; skipping.")
        return

    archive = download_one_of(CIFAR10_URLS, ARCHIVE_ROOT / "cifar-10-python.tar.gz")
    labels = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    per_label = {label: 0 for label in labels}
    reset_dir(images_dir)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmp_dir)
        batch_files = sorted((tmp_dir / "cifar-10-batches-py").glob("data_batch_*")) + [tmp_dir / "cifar-10-batches-py" / "test_batch"]
        for batch_file in batch_files:
            split = "test" if batch_file.name == "test_batch" else "train"
            with batch_file.open("rb") as f:
                batch = pickle.load(f, encoding="bytes")
            data = batch[b"data"]
            batch_labels = batch[b"labels"]
            for row_index, (row, label_index) in enumerate(zip(data, batch_labels)):
                label = labels[label_index]
                label_dir = images_dir / label
                label_dir.mkdir(parents=True, exist_ok=True)
                rgb = cifar_row_to_rgb(row)
                filename = f"cifar10_{split}_{label}_{per_label[label]:05d}_{row_index:05d}.png"
                write_png(label_dir / filename, 32, 32, rgb, color_type=2)
                per_label[label] += 1
            print(f"CIFAR-10 {batch_file.name} exported: {len(data)} images.")

    write_metadata(target, {
        "id": "cifar10-500",
        "sourceUrl": CIFAR10_URLS,
        "sampleCount": sum(per_label.values()),
        "labels": labels,
        "layout": "images/{label}/*.png",
    })
    print("CIFAR-10 full dataset ready.")


def download_iris() -> None:
    target = DATA_ROOT / "iris"
    csv_path = target / "iris.csv"
    if csv_path.exists():
        print("Iris CSV already exists; skipping.")
        return

    target.mkdir(parents=True, exist_ok=True)
    raw = download(IRIS_URL, ARCHIVE_ROOT / "iris.data").read_text(encoding="utf-8")
    rows = [line.split(",") for line in raw.splitlines() if line.strip()]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sepal_length", "sepal_width", "petal_length", "petal_width", "label"])
        for row in rows:
            label = row[4].replace("Iris-", "")
            writer.writerow(row[:4] + [label])

    write_metadata(target, {
        "id": "iris",
        "sourceUrl": IRIS_URL,
        "sampleCount": len(rows),
        "labels": sorted({row[4].replace("Iris-", "") for row in rows}),
        "layout": "iris.csv",
    })
    print("Iris dataset ready.")


def generate_points() -> None:
    target = DATA_ROOT / "points-2d"
    csv_path = target / "points.csv"
    if csv_path.exists():
        print("2D points CSV already exists; skipping.")
        return

    target.mkdir(parents=True, exist_ok=True)
    random.seed(20260427)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label"])
        for index in range(300):
            label = "class A" if index < 150 else "class B"
            center_x = -0.35 if label == "class A" else 0.35
            center_y = -0.15 if label == "class A" else 0.15
            angle = random.random() * math.tau
            radius = random.gauss(0.0, 0.18)
            x = center_x + math.cos(angle) * radius + random.gauss(0.0, 0.06)
            y = center_y + math.sin(angle) * radius + random.gauss(0.0, 0.06)
            writer.writerow([f"{x:.5f}", f"{y:.5f}", label])

    write_metadata(target, {
        "id": "points-2d",
        "sourceUrl": "generated",
        "sampleCount": 300,
        "labels": ["class A", "class B"],
        "layout": "points.csv",
    })
    print("2D points dataset ready.")


def write_catalog() -> None:
    datasets = []
    for metadata_path in sorted(DATA_ROOT.glob("*/metadata.json")):
        datasets.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    (DATA_ROOT / "catalog.json").write_text(
        json.dumps({"datasets": datasets}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def download(url: str, target: Path) -> Path:
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(target)
    return target


def download_one_of(urls: list[str], target: Path) -> Path:
    if target.exists() and target.stat().st_size > 0:
        return target
    errors = []
    for url in urls:
        try:
            return download(url, target)
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            tmp = target.with_suffix(target.suffix + ".tmp")
            if tmp.exists():
                tmp.unlink()
            print(f"Download failed, trying next source: {exc}")
    raise RuntimeError("All download sources failed:\n" + "\n".join(errors))


def export_mnist_split(
    image_archive: Path,
    label_archive: Path,
    images_dir: Path,
    split: str,
    per_label: dict[str, int],
) -> int:
    with gzip.open(image_archive, "rb") as f:
        magic, total, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Invalid MNIST image file.")
        image_bytes = f.read()
    with gzip.open(label_archive, "rb") as f:
        magic, label_total = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Invalid MNIST label file.")
        labels = f.read()
    if total != label_total:
        raise ValueError("MNIST image/label count mismatch.")

    image_size = rows * cols
    for index, label_value in enumerate(labels):
        label = str(label_value)
        label_dir = images_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        start = index * image_size
        pixels = image_bytes[start:start + image_size]
        filename = f"mnist_{split}_{label}_{per_label[label]:05d}_{index:05d}.png"
        write_png(label_dir / filename, cols, rows, pixels, color_type=0)
        per_label[label] += 1
    return total


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob(pattern))


def write_metadata(target: Path, metadata: dict) -> None:
    target.mkdir(parents=True, exist_ok=True)
    (target / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def cifar_row_to_rgb(row) -> bytes:
    raw = bytes(row)
    red = raw[0:1024]
    green = raw[1024:2048]
    blue = raw[2048:3072]
    rgb = bytearray()
    for i in range(1024):
        rgb.extend((red[i], green[i], blue[i]))
    return bytes(rgb)


def write_png(path: Path, width: int, height: int, pixels: bytes, color_type: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = 1 if color_type == 0 else 3
    stride = width * channels
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        start = y * stride
        rows.extend(pixels[start:start + stride])

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)))
    png.extend(chunk(b"IDAT", zlib.compress(bytes(rows), level=6)))
    png.extend(chunk(b"IEND", b""))
    path.write_bytes(bytes(png))


if __name__ == "__main__":
    sys.exit(main())
