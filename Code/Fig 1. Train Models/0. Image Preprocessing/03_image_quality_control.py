"""Quality control filtering of image tiles."""

import os
import shutil
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import tifffile
from tqdm import tqdm


args = Namespace(
    input_dir=Path("path/to/cropped_images"),
    output_dir=Path("path/to/rejected_images"),
    min_std_sum=3500.0,
    min_signal_sum=8000.0,
    min_laplacian_var=50000.0,
    max_saturation_percent=0.5,
    min_saturation_percent=10.0,
    bg_threshold=4000.0,
    max_bg_fraction=0.65,
)


def check_quality(img, args):
    if img.ndim == 2:
        img = img[..., np.newaxis]

    h, w, c = img.shape

    std_sum = sum(
        np.std(img[..., i].astype(np.float32))
        for i in range(c)
    )
    if std_sum < args.min_std_sum:
        return True, "Low variation"

    top_fraction = int(h * w * 0.15)
    total_signal = 0

    for i in range(c):
        channel = img[..., i].ravel()
        if top_fraction > 0:
            partitioned = np.partition(channel, -top_fraction)
            total_signal += np.sum(partitioned[-top_fraction:])

    if total_signal < args.min_signal_sum:
        return True, "Weak signal"

    max_channel = np.max(img, axis=2).astype(np.float64)
    laplacian_var = cv2.Laplacian(
        max_channel,
        cv2.CV_64F,
    ).var()

    if laplacian_var < args.min_laplacian_var:
        return True, "Blurry"

    scaled_img = np.zeros_like(img, dtype=np.float32)
    target_max = 65535.0
    min_std_threshold = 655.0

    for i in range(c):
        channel = img[..., i].astype(np.float32)

        if np.std(channel) < min_std_threshold:
            scaled_channel = channel
        else:
            min_cutoff = np.percentile(
                channel,
                args.min_saturation_percent,
            )
            max_cutoff = np.percentile(
                channel,
                100 - args.max_saturation_percent,
            )

            if max_cutoff <= min_cutoff:
                scaled_channel = np.zeros_like(channel)
            else:
                scaled_channel = (
                    (channel - min_cutoff)
                    * target_max
                    / (max_cutoff - min_cutoff)
                )

        scaled_img[..., i] = np.clip(
            scaled_channel,
            0,
            target_max,
        )

    max_projection = np.max(
        scaled_img,
        axis=2,
    )

    background_fraction = (
        1
        - np.count_nonzero(
            max_projection > args.bg_threshold
        )
        / (h * w)
    )

    if background_fraction > args.max_bg_fraction:
        return True, "Too empty"

    return False, "Pass"


def run_filtering(args):
    files = []

    for root, _, filenames in os.walk(args.input_dir):
        if Path(root).resolve().is_relative_to(
            args.output_dir.resolve()
        ):
            continue

        for filename in filenames:
            if filename.lower().endswith((".tif", ".tiff")):
                files.append(Path(root) / filename)

    for file_path in tqdm(files, desc="Quality control"):
        img = tifffile.imread(file_path)
        rejected, _ = check_quality(img, args)

        if rejected:
            relative_path = file_path.relative_to(args.input_dir)
            destination = args.output_dir / relative_path

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            shutil.move(
                str(file_path),
                str(destination),
            )


if __name__ == "__main__":
    run_filtering(args)
