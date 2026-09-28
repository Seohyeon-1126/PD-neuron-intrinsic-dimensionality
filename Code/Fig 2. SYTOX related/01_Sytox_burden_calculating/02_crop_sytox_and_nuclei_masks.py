"""Crop SYTOX images and nuclei masks at QC-passed tile coordinates."""

import re
from pathlib import Path

import tifffile
from tqdm import tqdm


QC_DIR = Path("path/to/qc_passed_tiles")
MASK_DIR = Path("path/to/nuclei_masks")
SYTOX_DIR = Path("path/to/sytox_mip")
OUTPUT_DIR = Path("path/to/sytox_analysis")
GROUPS = ["group_1", "group_2"]

PATCH_SIZE = 128


def parse_tile_name(filename):
    match = re.search(r"_x(\d+)_y(\d+)", filename)
    if not match:
        return None

    x, y = map(int, match.groups())
    base = filename[:match.start()]

    plate_match = re.match(
        r"^(\d+_[a-zA-Z]\d+c\d+f\d+)",
        base,
    )

    if not plate_match:
        return None

    return plate_match.group(1), x, y


def find_image(directory, pattern):
    matches = list(directory.glob(pattern))
    return matches[0] if matches else None


def crop_image(image, x, y):
    return image[
        y:y + PATCH_SIZE,
        x:x + PATCH_SIZE,
    ]


def run():
    for group in GROUPS:
        qc_dir = QC_DIR / group
        mask_dir = MASK_DIR / group / "mask"
        sytox_dir = SYTOX_DIR / group

        mask_output = OUTPUT_DIR / group / "stardist_mask"
        sytox_output = OUTPUT_DIR / group / "cytoxgreen"

        mask_output.mkdir(parents=True, exist_ok=True)
        sytox_output.mkdir(parents=True, exist_ok=True)

        qc_files = sorted(
            list(qc_dir.glob("*.tif"))
            + list(qc_dir.glob("*.tiff"))
        )

        mask_cache = {}
        sytox_cache = {}

        for qc_file in tqdm(qc_files, desc=group):
            parsed = parse_tile_name(qc_file.name)
            if parsed is None:
                continue

            plate_position, x, y = parsed

            if plate_position not in mask_cache:
                path = find_image(
                    mask_dir,
                    f"{plate_position}_*_mask.tif",
                )
                mask_cache[plate_position] = (
                    tifffile.imread(path) if path else None
                )

            if plate_position not in sytox_cache:
                path = find_image(
                    sytox_dir,
                    f"{plate_position}_*Cytox*.tif",
                )
                sytox_cache[plate_position] = (
                    tifffile.imread(path) if path else None
                )

            base = qc_file.stem

            mask = mask_cache[plate_position]
            if mask is not None:
                patch = crop_image(mask, x, y)
                if patch.shape[:2] == (PATCH_SIZE, PATCH_SIZE):
                    tifffile.imwrite(
                        mask_output / f"{base}_mask.tif",
                        patch,
                    )

            sytox = sytox_cache[plate_position]
            if sytox is not None:
                patch = crop_image(sytox, x, y)
                if patch.shape[:2] == (PATCH_SIZE, PATCH_SIZE):
                    tifffile.imwrite(
                        sytox_output / f"{base}_cytox.tif",
                        patch,
                    )


if __name__ == "__main__":
    run()
