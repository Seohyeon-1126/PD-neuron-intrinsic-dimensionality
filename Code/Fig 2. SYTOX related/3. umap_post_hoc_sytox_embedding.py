"""UMAP embedding of SYTOX-matched disease latent representations."""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import umap


LATENT_DIR = Path("path/to/latent_vectors")
SYTOX_CSV = Path("path/to/per_image_sytox_burden.csv")
OUTPUT_DIR = Path("path/to/umap_output")

DISEASE_GROUPS = ["GBA", "PINK1", "SNCA"]
TARGET_COLUMN = "intensity_per_pixel_mean_3std"

COMBINED_N_NEIGHBORS = 15
SUBTYPE_N_NEIGHBORS = 30
MIN_DIST = 0.1


def normalize_tile_id(value):
    value = os.path.basename(str(value))
    value = re.sub(
        r"\.(tif|tiff|png|jpg|jpeg|npy|csv)$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"_(mask|cytox)$",
        "",
        value,
        flags=re.IGNORECASE,
    )


def load_sytox():
    df = pd.read_csv(SYTOX_CSV)

    required = {"filename", TARGET_COLUMN}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df["tile_id"] = df[
        "filename"
    ].map(normalize_tile_id)

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    return (
        df.dropna(
            subset=["tile_id", TARGET_COLUMN]
        )
        .groupby("tile_id")[TARGET_COLUMN]
        .mean()
        .to_dict()
    )


def load_group(group, tile_to_sytox):
    latent = np.load(
        LATENT_DIR / f"{group}_latent_raw.npy",
        mmap_mode="r",
    )
    filenames = np.load(
        LATENT_DIR / f"{group}_filenames.npy",
        allow_pickle=True,
    ).astype(str)

    if len(latent) != len(filenames):
        raise ValueError(
            f"{group}: latent and filename counts differ."
        )

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

    valid = np.isfinite(sytox)

    return (
        np.asarray(
            latent[valid],
            dtype=np.float32,
        ),
        filenames[valid],
        tile_ids[valid],
        sytox[valid],
    )


def run_umap(latent, n_neighbors):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=MIN_DIST,
        metric="euclidean",
    )
    return reducer.fit_transform(latent)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tile_to_sytox = load_sytox()
    group_data = {}

    for group in DISEASE_GROUPS:
        group_data[group] = load_group(
            group,
            tile_to_sytox,
        )

    combined_latent = np.concatenate(
        [
            group_data[group][0]
            for group in DISEASE_GROUPS
        ],
        axis=0,
    )

    combined_group = np.concatenate(
        [
            np.repeat(
                group,
                len(group_data[group][0]),
            )
            for group in DISEASE_GROUPS
        ]
    )

    combined_filenames = np.concatenate(
        [
            group_data[group][1]
            for group in DISEASE_GROUPS
        ]
    )

    combined_tile_ids = np.concatenate(
        [
            group_data[group][2]
            for group in DISEASE_GROUPS
        ]
    )

    combined_sytox = np.concatenate(
        [
            group_data[group][3]
            for group in DISEASE_GROUPS
        ]
    )

    combined_embedding = run_umap(
        combined_latent,
        COMBINED_N_NEIGHBORS,
    )

    combined_df = pd.DataFrame(
        {
            "filename": combined_filenames,
            "tile_id": combined_tile_ids,
            "group": combined_group,
            "sytox": combined_sytox,
            "umap_1": combined_embedding[:, 0],
            "umap_2": combined_embedding[:, 1],
        }
    )

    combined_df.to_csv(
        OUTPUT_DIR / "combined_disease_umap.csv",
        index=False,
    )

    for group in DISEASE_GROUPS:
        latent, filenames, tile_ids, sytox = (
            group_data[group]
        )

        embedding = run_umap(
            latent,
            SUBTYPE_N_NEIGHBORS,
        )

        pd.DataFrame(
            {
                "filename": filenames,
                "tile_id": tile_ids,
                "group": group,
                "sytox": sytox,
                "umap_1": embedding[:, 0],
                "umap_2": embedding[:, 1],
            }
        ).to_csv(
            OUTPUT_DIR / f"{group}_umap.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
