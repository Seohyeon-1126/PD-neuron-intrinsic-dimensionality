"""Calculate per-image SYTOX cell-death burden."""

import csv
import re
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.restoration import rolling_ball
from tqdm import tqdm


DATA_DIR = Path("path/to/sytox_analysis")
OUTPUT_CSV = Path("path/to/per_image_sytox_burden.csv")

GROUPS = [
    "Control_C4",
    "Control_C18",
    "Control_C19",
    "GBA",
    "SNCA",
    "PINK1",
]

THRESHOLDS = {
    "mean_3std": 24712.982763,
}

GAUSSIAN_SIGMA = 1.0
ROLLING_BALL_RADIUS = 25
MIN_NUCLEUS_FRACTION = 0.25

ORIGINAL_SIZE = 1080
PATCH_SIZE = 128
STRIDE = 128
EXCLUDE_EDGE_PATCHES = True


def class_label(group):
    return "Control" if group.startswith("Control") else group


def last_crop_coordinate():
    return (
        (ORIGINAL_SIZE - PATCH_SIZE)
        // STRIDE
        * STRIDE
    )


def is_edge_patch(filename):
    match = re.search(
        r"_x(\d+)_y(\d+)",
        filename,
    )

    if match is None:
        return True

    x, y = map(int, match.groups())
    max_coord = last_crop_coordinate()

    return (
        x == 0
        or x == max_coord
        or y == 0
        or y == max_coord
    )


def preprocess_sytox(image):
    image = image.astype(np.float64)

    image = gaussian_filter(
        image,
        sigma=GAUSSIAN_SIGMA,
    )

    background = rolling_ball(
        image,
        radius=ROLLING_BALL_RADIUS,
    )

    return np.maximum(
        image - background,
        0,
    )


def find_sytox(mask_path, sytox_dir):
    stem = re.sub(
        r"_mask$",
        "",
        mask_path.stem,
        flags=re.IGNORECASE,
    )

    for suffix in (".tif", ".tiff"):
        path = sytox_dir / f"{stem}_cytox{suffix}"

        if path.exists():
            return path

    return None


def clean_tile_id(mask_path):
    return re.sub(
        r"_mask$",
        "",
        mask_path.stem,
        flags=re.IGNORECASE,
    )


def quantify_group(group):
    mask_dir = (
        DATA_DIR
        / group
        / "QC"
        / "stardist_mask"
    )
    sytox_dir = (
        DATA_DIR
        / group
        / "QC"
        / "cytoxgreen"
    )

    if not mask_dir.exists() or not sytox_dir.exists():
        return []

    mask_files = sorted(
        list(mask_dir.glob("*_mask.tif"))
        + list(mask_dir.glob("*_mask.tiff"))
    )

    if EXCLUDE_EDGE_PATCHES:
        mask_files = [
            path
            for path in mask_files
            if not is_edge_patch(path.name)
        ]

    min_nucleus_pixels = int(
        PATCH_SIZE
        * PATCH_SIZE
        * MIN_NUCLEUS_FRACTION
    )

    results = []

    for mask_path in tqdm(mask_files, desc=group):
        sytox_path = find_sytox(
            mask_path,
            sytox_dir,
        )

        if sytox_path is None:
            continue

        mask = np.squeeze(
            tifffile.imread(mask_path)
        )
        sytox = np.squeeze(
            tifffile.imread(sytox_path)
        )

        if mask.ndim == 3:
            mask = mask[..., 0]

        if sytox.ndim == 3:
            sytox = sytox[..., 0]

        if (
            mask.ndim != 2
            or sytox.ndim != 2
            or mask.shape != sytox.shape
        ):
            continue

        nucleus_mask = mask > 0
        nucleus_pixels = int(
            nucleus_mask.sum()
        )

        if nucleus_pixels < min_nucleus_pixels:
            continue

        processed = preprocess_sytox(sytox)
        values = processed[
            nucleus_mask
        ].astype(np.float64)

        total_intensity = float(
            values.sum()
        )

        result = {
            "filename": clean_tile_id(mask_path),
            "folder": group,
            "class": class_label(group),
            "total_nucleus_pixels": nucleus_pixels,
            "nucleus_fraction": (
                nucleus_pixels
                / (PATCH_SIZE ** 2)
            ),
            "total_intensity": total_intensity,
        }

        for name, threshold in THRESHOLDS.items():
            positive = values >= threshold
            positive_pixels = int(
                positive.sum()
            )
            positive_intensity = float(
                values[positive].sum()
            )

            result[f"threshold_{name}"] = threshold
            result[f"rate_{name}"] = (
                positive_pixels
                / nucleus_pixels
            )
            result[f"intensity_rate_{name}"] = (
                positive_intensity
                / total_intensity
                if total_intensity > 0
                else 0.0
            )
            result[f"intensity_per_pixel_{name}"] = (
                positive_intensity
                / nucleus_pixels
            )

        results.append(result)

    return results


def main():
    results = []

    for group in GROUPS:
        results.extend(
            quantify_group(group)
        )

    if not results:
        raise RuntimeError(
            "No valid images were found."
        )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
