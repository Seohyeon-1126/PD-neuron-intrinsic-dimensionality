"""Crop RGB MIP images into non-overlapping tiles."""

from pathlib import Path

import tifffile
from tqdm import tqdm


INPUT_DIR = Path("path/to/mip_images")
OUTPUT_DIR = Path("path/to/cropped_images")

PATCH_SIZE = 128
STRIDE = 128


def is_rgb_mip_file(path):
    name = path.name.lower()
    return (
        "composite_rgb" in name
        and "mask" not in name
        and "cytox" not in name
        and "ch2" not in name
    )


def crop_images(input_dir, output_dir):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tif_files = sorted(
        [
            path
            for path in list(input_dir.glob("*.tif"))
            + list(input_dir.glob("*.tiff"))
            if is_rgb_mip_file(path)
        ]
    )

    for tif_path in tqdm(
        tif_files,
        desc="Cropping images",
    ):
        img = tifffile.imread(tif_path)

        if img.ndim < 2:
            continue

        height, width = img.shape[:2]

        for y in range(
            0,
            height - PATCH_SIZE + 1,
            STRIDE,
        ):
            for x in range(
                0,
                width - PATCH_SIZE + 1,
                STRIDE,
            ):
                patch = img[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE,
                ]

                if patch.shape[:2] != (
                    PATCH_SIZE,
                    PATCH_SIZE,
                ):
                    continue

                output_path = (
                    output_dir
                    / f"{tif_path.stem}_x{x}_y{y}.tif"
                )

                tifffile.imwrite(
                    output_path,
                    patch,
                )


if __name__ == "__main__":
    crop_images(
        INPUT_DIR,
        OUTPUT_DIR,
    )
