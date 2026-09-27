"""UMAP visualization of raw latent representations."""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import umap


LATENT_DIR = Path("path/to/latent_vectors")
OUTPUT_CSV = Path("path/to/umap_coordinates.csv")

CONTROL_GROUPS = [
    "Control_GBA_C19",
    "Control_SNCA_C19",
    "Control_C4",
    "Control_C18",
]
DISEASE_GROUPS = ["GBA", "PINK1", "SNCA"]
ALL_GROUPS = CONTROL_GROUPS + DISEASE_GROUPS

SAMPLES_PER_CLASS = 20000
SAMPLES_PER_CONTROL_GROUP = (
    SAMPLES_PER_CLASS // len(CONTROL_GROUPS)
)

UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "euclidean"


def normalize_tile_id(value):
    value = os.path.basename(str(value))
    value = re.sub(
        r"\.(npy|tif|tiff|png|jpg|jpeg)$",
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


def load_group(group):
    latent_path = (
        LATENT_DIR / f"{group}_latent_raw.npy"
    )
    filename_path = (
        LATENT_DIR / f"{group}_filenames.npy"
    )

    latent = np.load(latent_path)
    filenames = np.load(
        filename_path,
        allow_pickle=True,
    )

    if latent.ndim != 2:
        raise ValueError(
            f"{group}: latent array must be 2D."
        )

    if len(latent) != len(filenames):
        raise ValueError(
            f"{group}: latent and filename counts differ."
        )

    tile_ids = np.asarray(
        [
            normalize_tile_id(name)
            for name in filenames
        ]
    )

    return latent, tile_ids


def sample_group(latent, tile_ids, n_samples):
    n_samples = min(
        n_samples,
        len(latent),
    )

    indices = np.random.choice(
        len(latent),
        size=n_samples,
        replace=False,
    )

    return (
        np.asarray(
            latent[indices],
            dtype=np.float32,
        ),
        tile_ids[indices],
    )


def load_sampled_latents():
    latent_parts = []
    metadata = []

    for group in ALL_GROUPS:
        latent, tile_ids = load_group(group)

        if group in CONTROL_GROUPS:
            n_samples = SAMPLES_PER_CONTROL_GROUP
            class_name = "Control"
        else:
            n_samples = SAMPLES_PER_CLASS
            class_name = group

        sampled_latent, sampled_ids = sample_group(
            latent,
            tile_ids,
            n_samples,
        )

        latent_parts.append(sampled_latent)

        metadata.extend(
            {
                "raw_group": group,
                "class": class_name,
                "tile_id": tile_id,
            }
            for tile_id in sampled_ids
        )

    return (
        np.concatenate(
            latent_parts,
            axis=0,
        ),
        pd.DataFrame(metadata),
    )


def calculate_umap(latent):
    if not np.all(np.isfinite(latent)):
        raise ValueError(
            "Latent representations contain NaN or inf."
        )

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        init="spectral",
        low_memory=True,
    )

    return reducer.fit_transform(latent)


def main():
    latent, metadata = load_sampled_latents()
    embedding = calculate_umap(latent)

    metadata["umap_1"] = embedding[:, 0]
    metadata["umap_2"] = embedding[:, 1]

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
