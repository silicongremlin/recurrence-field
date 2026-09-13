"""
image_analysis.py

Local image-analysis bridge for recurrence field outputs.

Purpose:
- Analyze generated recurrence field PNGs.
- Extract lightweight visual/statistical features.
- Produce CSV summaries.
- Optionally compute pairwise distances between images.
- Provide a measurement layer without requiring ML infrastructure.

Usage:
    python image_analysis.py --input outputs
    python image_analysis.py --input outputs --pairwise
    python image_analysis.py --input outputs --out analysis_results
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class ImageFeatures:
    file: str
    width: int
    height: int

    mean: float
    std: float
    min_value: float
    max_value: float
    dynamic_range: float

    p05: float
    p25: float
    p50: float
    p75: float
    p95: float

    entropy: float
    edge_energy: float
    gradient_mean: float
    gradient_std: float

    radial_mean_inner: float
    radial_mean_middle: float
    radial_mean_outer: float

    quadrant_mean_ul: float
    quadrant_mean_ur: float
    quadrant_mean_ll: float
    quadrant_mean_lr: float


def load_image_grayscale(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    arr = np.asarray(image, dtype=np.float64) / 255.0
    return arr


def entropy(arr: np.ndarray, bins: int = 256) -> float:
    hist, _ = np.histogram(arr, bins=bins, range=(0.0, 1.0), density=False)
    probs = hist.astype(np.float64)
    probs = probs / probs.sum()

    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def gradient_features(arr: np.ndarray) -> tuple[float, float, float]:
    gy, gx = np.gradient(arr)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    edge_energy = float(np.mean(mag ** 2))
    gradient_mean = float(np.mean(mag))
    gradient_std = float(np.std(mag))

    return edge_energy, gradient_mean, gradient_std


def radial_features(arr: np.ndarray) -> tuple[float, float, float]:
    h, w = arr.shape
    yy, xx = np.indices((h, w))

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_norm = r / r.max()

    inner = arr[r_norm <= 0.33]
    middle = arr[(r_norm > 0.33) & (r_norm <= 0.66)]
    outer = arr[r_norm > 0.66]

    return (
        float(np.mean(inner)),
        float(np.mean(middle)),
        float(np.mean(outer)),
    )


def quadrant_features(arr: np.ndarray) -> tuple[float, float, float, float]:
    h, w = arr.shape
    h2 = h // 2
    w2 = w // 2

    ul = arr[:h2, :w2]
    ur = arr[:h2, w2:]
    ll = arr[h2:, :w2]
    lr = arr[h2:, w2:]

    return (
        float(np.mean(ul)),
        float(np.mean(ur)),
        float(np.mean(ll)),
        float(np.mean(lr)),
    )


def analyze_image(path: Path) -> ImageFeatures:
    arr = load_image_grayscale(path)

    p05, p25, p50, p75, p95 = np.percentile(arr, [5, 25, 50, 75, 95])
    edge_energy, gradient_mean, gradient_std = gradient_features(arr)
    r_inner, r_middle, r_outer = radial_features(arr)
    q_ul, q_ur, q_ll, q_lr = quadrant_features(arr)

    return ImageFeatures(
        file=str(path),
        width=arr.shape[1],
        height=arr.shape[0],

        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        min_value=float(np.min(arr)),
        max_value=float(np.max(arr)),
        dynamic_range=float(np.max(arr) - np.min(arr)),

        p05=float(p05),
        p25=float(p25),
        p50=float(p50),
        p75=float(p75),
        p95=float(p95),

        entropy=entropy(arr),
        edge_energy=edge_energy,
        gradient_mean=gradient_mean,
        gradient_std=gradient_std,

        radial_mean_inner=r_inner,
        radial_mean_middle=r_middle,
        radial_mean_outer=r_outer,

        quadrant_mean_ul=q_ul,
        quadrant_mean_ur=q_ur,
        quadrant_mean_ll=q_ll,
        quadrant_mean_lr=q_lr,
    )


def image_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    return sorted(
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_feature_csv(features: list[ImageFeatures], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not features:
        raise ValueError("No image features to write.")

    rows = [asdict(f) for f in features]

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resized_vector(path: Path, size: int = 128) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size))
    arr = np.asarray(img, dtype=np.float64) / 255.0

    # Normalize so distance compares structure more than absolute brightness.
    arr = arr - arr.mean()
    std = arr.std()
    if std > 0:
        arr = arr / std

    return arr.reshape(-1)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0

    similarity = float(np.dot(a, b) / denom)
    return float(1.0 - similarity)


def write_pairwise_csv(paths: list[Path], out_path: Path, vector_size: int = 128) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vectors = {path: resized_vector(path, size=vector_size) for path in paths}

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_a",
                "file_b",
                "cosine_distance",
            ],
        )
        writer.writeheader()

        for i, path_a in enumerate(paths):
            for path_b in paths[i + 1:]:
                writer.writerow(
                    {
                        "file_a": str(path_a),
                        "file_b": str(path_b),
                        "cosine_distance": cosine_distance(vectors[path_a], vectors[path_b]),
                    }
                )


def bridge_summary(features: list[ImageFeatures]) -> str:
    """
    Produces a simple measurement summary:
    not interpretation-as-truth, just a compact measurement layer.
    """

    if not features:
        return "No images analyzed."

    entropy_values = np.array([f.entropy for f in features])
    std_values = np.array([f.std for f in features])
    edge_values = np.array([f.edge_energy for f in features])

    return "\n".join(
        [
            "RECURRENCE IMAGE ANALYSIS SUMMARY",
            "",
            f"images_analyzed: {len(features)}",
            f"entropy_mean: {entropy_values.mean():.6f}",
            f"entropy_std: {entropy_values.std():.6f}",
            f"field_std_mean: {std_values.mean():.6f}",
            f"field_std_std: {std_values.std():.6f}",
            f"edge_energy_mean: {edge_values.mean():.8f}",
            f"edge_energy_std: {edge_values.std():.8f}",
            "",
            "note:",
            "  These measurements describe rendered image structure only.",
            "  They do not prove recurrence behavior, causality, or topology by themselves.",
            "  Use them as a measurement layer between visual inspection and later experiment design.",
        ]
    )


def analyze_directory(input_path: Path, out_dir: Path, pairwise: bool = False, vector_size: int = 128) -> None:
    """Analyze one image or a directory of images and write measurement artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = image_paths(input_path)
    if not paths:
        raise SystemExit(f"No image files found at: {input_path}")

    features = [analyze_image(path) for path in paths]
    feature_csv = out_dir / "image_features.csv"
    write_feature_csv(features, feature_csv)

    summary = bridge_summary(features)
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    print(summary)
    print(f"features_csv: {feature_csv}")
    print(f"summary_txt: {summary_path}")

    if pairwise:
        pairwise_csv = out_dir / "pairwise_distances.csv"
        write_pairwise_csv(paths, pairwise_csv, vector_size=vector_size)
        print(f"pairwise_csv: {pairwise_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze recurrence field output images."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Image file or directory containing recurrence output images.",
    )
    parser.add_argument(
        "--out",
        default="analysis_results",
        help="Output directory for analysis CSV and summary files.",
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Also compute pairwise cosine distances over resized grayscale images.",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=128,
        help="Resize dimension for pairwise image vectors.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)
    analyze_directory(input_path, out_dir, args.pairwise, args.vector_size)


if __name__ == "__main__":
    main()
