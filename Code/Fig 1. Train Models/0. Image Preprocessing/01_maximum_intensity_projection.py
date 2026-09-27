"""Maximum-intensity projection and channel composition."""

import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm


EXPERIMENTS = {
    "group_name": {
        "rows": [5, 6, 7],
        "input_dirs": [
            Path("path/to/image_folder_1"),
            Path("path/to/image_folder_2"),
            Path("path/to/image_folder_3"),
        ],
    },
}

MIP_OUTPUT_DIR = Path("path/to/mip_output")
SYTOX_OUTPUT_DIR = Path("path/to/sytox_output")

COLS_RANGE = range(2, 12)
Z_STACK_COUNT = 5


@dataclass(frozen=True)
class ImageMetadata:
    group_name: str
    folder_id: str
    row: int
    col: int
    fraction: int
    plane: int
    channel: int
    full_path: Path

    @property
    def group_key(self):
        return (
            self.group_name,
            self.folder_id,
            self.row,
            self.col,
            self.fraction,
            self.channel,
        )


def extract_folder_id(path):
    parent = path.parent.name
    match = re.match(r"(\d+)", parent)
    return match.group(1) if match else "Unknown"


def parse_files(group_name, config):
    pattern = re.compile(r"r(\d+)c(\d+)f(\d+)p(\d+)-ch(\d+)")
    deduplicated = {}

    for folder in config["input_dirs"]:
        if not folder.exists():
            continue

        folder_id = extract_folder_id(folder)

        for filename in tqdm(os.listdir(folder), desc=f"Scanning {folder_id}"):
            match = pattern.search(filename)
            if not match:
                continue

            row, col, fraction, plane, channel = map(int, match.groups())

            if row not in config["rows"] or col not in COLS_RANGE:
                continue

            key = (
                folder_id,
                row,
                col,
                fraction,
                plane,
                channel,
            )

            deduplicated[key] = ImageMetadata(
                group_name=group_name,
                folder_id=folder_id,
                row=row,
                col=col,
                fraction=fraction,
                plane=plane,
                channel=channel,
                full_path=folder / filename,
            )

    return list(deduplicated.values())


def calculate_mip(images):
    grouped = defaultdict(list)

    for image in images:
        grouped[image.group_key].append(image)

    results = defaultdict(dict)

    for key, stack in tqdm(grouped.items(), desc="Calculating MIP"):
        _, folder_id, row, col, fraction, channel = key

        stack = sorted(
            stack,
            key=lambda x: x.plane,
        )[:Z_STACK_COUNT]

        if len(stack) < Z_STACK_COUNT:
            continue

        image_stack = np.asarray(
            [tifffile.imread(x.full_path) for x in stack]
        )

        mip = np.max(
            image_stack,
            axis=0,
        ).astype(np.uint16)

        results[
            (folder_id, row, col, fraction)
        ][channel] = mip

    return results


def save_images(results, group_name):
    rgb_dir = MIP_OUTPUT_DIR / group_name
    sytox_dir = SYTOX_OUTPUT_DIR / group_name

    rgb_dir.mkdir(parents=True, exist_ok=True)
    sytox_dir.mkdir(parents=True, exist_ok=True)

    for (folder_id, row, col, fraction), channels in tqdm(
        results.items(),
        desc="Saving",
    ):
        prefix = (
            f"{folder_id}_"
            f"r{row:02d}c{col:02d}f{fraction:02d}"
        )

        if all(channel in channels for channel in [1, 3, 4]):
            rgb = np.stack(
                [
                    channels[3],
                    channels[4],
                    channels[1],
                ],
                axis=-1,
            )

            tifffile.imwrite(
                rgb_dir / f"{prefix}_Composite_RGB.tif",
                rgb,
            )

        if 2 in channels:
            tifffile.imwrite(
                sytox_dir / f"{prefix}_MIP_ch2_Cytox.tif",
                channels[2],
            )


def main():
    for group_name, config in EXPERIMENTS.items():
        images = parse_files(
            group_name,
            config,
        )

        if not images:
            continue

        mip_results = calculate_mip(images)

        save_images(
            mip_results,
            group_name,
        )


if __name__ == "__main__":
    main()
