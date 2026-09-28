"""PHATE embedding with disease-specific SYTOX annotation."""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import phate


LATENT_DIR = Path("path/to/latent_vectors")
SYTOX_CSV = Path("path/to/per_image_sytox_burden.csv")
OUTPUT_CSV = Path("path/to/phate_coordinates.csv")

SOURCE_GROUPS = [
    "Control_GBA_C19",
    "Control_SNCA_C19",
    "Control_C4",
    "Control_C18",
    "GBA",
    "PINK1",
    "SNCA",
]

MAJOR_GROUP_MAP = {
    "Control_GBA_C19": "Control",
    "Control_SNCA_C19": "Control",
    "Control_C4": "Control",
    "Control_C18": "Control",
    "GBA": "GBA",
    "PINK1": "PINK1",
    "SNCA": "SNCA",
}

SYTOX_COLUMN = "intensity_per_pixel_mean_3std"
MAX_POINTS_PER_SOURCE_GROUP = 5000

USE_L2_NORM = True
PHATE_KNN = 15
PHATE_DECAY = 40
PHATE_T = "auto"


def normalize_tile_id(value):
    value = os.path.basename(str(value))
    value = re.sub(
        r"\.(tif|tiff|png|jpg|jpeg|npy|csv)$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"_mask$",
        "",
        value,
        flags=re.IGNORECASE,
    )


def l2_normalize(latent, eps=1e-12):
    latent = np.asarray(
        latent,
        dtype=np.float32,
    )
    norms = np.linalg.norm(
        latent,
        axis=1,
        keepdims=True,
    )
    return latent / np.maximum(
        norms,
        eps,
    )


def load_group(group):
    latent_path = (
        LATENT_DIR / f"{group}_latent_raw.npy"
    )
    filename_path = (
        LATENT_DIR / f"{group}_filenames.npy"
    )

    latent = np.load(
        latent_path,
        mmap_mode="r",
    )
    filenames = np.load(
        filename_path,
        allow_pickle=True,
    ).astype(str)

    if len(latent) != len(filenames):
        raise ValueError(
            f"{group}: latent and filename counts differ."
        )

    return latent, filenames


def load_sytox():
    df = pd.read_csv(SYTOX_CSV)

    if "filename" not in df.columns:
        raise ValueError(
            'SYTOX CSV must contain "filename".'
        )

    if SYTOX_COLUMN not in df.columns:
        raise ValueError(
            f"Missing SYTOX column: {SYTOX_COLUMN}"
        )

    df["tile_id"] = df[
        "filename"
    ].apply(normalize_tile_id)

    df[SYTOX_COLUMN] = pd.to_numeric(
        df[SYTOX_COLUMN],
        errors="coerce",
    )

    return dict(
        zip(
            df["tile_id"],
            df[SYTOX_COLUMN],
        )
    )


def sample_indices(indices, max_points):
    if len(indices) <= max_points:
        return indices

    selected = np.random.choice(
        indices,
        size=max_points,
        replace=False,
    )
    return np.sort(selected)


def main():
    tile_to_sytox = load_sytox()

    latent_parts = []
    metadata_parts = []

    for group in SOURCE_GROUPS:
        latent, filenames = load_group(group)

        tile_ids = np.asarray(
            [
                normalize_tile_id(filename)
                for filename in filenames
            ]
        )

        sytox = np.asarray(
            [
                tile_to_sytox.get(
                    tile_id,
                    np.nan,
                )
                for tile_id in tile_ids
            ],
            dtype=float,
        )

        matched = np.flatnonzero(
            np.isfinite(sytox)
        )

        if len(matched) == 0:
            continue

        indices = sample_indices(
            matched,
            MAX_POINTS_PER_SOURCE_GROUP,
        )

        group_latent = np.asarray(
            latent[indices],
            dtype=np.float32,
        )

        if USE_L2_NORM:
            group_latent = l2_normalize(
                group_latent
            )

        latent_parts.append(group_latent)

        metadata_parts.append(
            pd.DataFrame(
                {
                    "filename": filenames[indices],
                    "tile_id": tile_ids[indices],
                    "source_group": group,
                    "major_group": MAJOR_GROUP_MAP[group],
                    "sytox": sytox[indices],
                }
            )
        )

    if not latent_parts:
        raise RuntimeError(
            "No SYTOX-matched latent representations found."
        )

    latent = np.concatenate(
        latent_parts,
        axis=0,
    )
    metadata = pd.concat(
        metadata_parts,
        ignore_index=True,
    )

    operator = phate.PHATE(
        n_components=2,
        knn=PHATE_KNN,
        decay=PHATE_DECAY,
        t=PHATE_T,
        n_jobs=-1,
        verbose=0,
    )

    embedding = operator.fit_transform(
        latent
    )

    metadata["phate_1"] = embedding[:, 0]
    metadata["phate_2"] = embedding[:, 1]

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    metadata.to_csv(
        OUTPUT_CSV,
        index=False,
    )


if __name__ == "__main__":
    main()
