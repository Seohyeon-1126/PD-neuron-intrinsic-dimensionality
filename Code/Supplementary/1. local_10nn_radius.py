"""Median local 10-NN radius across SYTOX windows."""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


SYTOX_CSV = Path("path/to/sytox_cell_death.csv")
TARGET_COLUMN = "intensity_per_pixel_mean_3std"

MODEL_NAME = "SupMoCo"
LATENT_DIR = Path("path/to/supmoco_latent_vectors")

DISEASES = ["GBA", "PINK1", "SNCA"]

N_WINDOWS = 30
WINDOW_FRAC = 0.10
MIN_WINDOW_N = 50

LOCAL_SCALE_K = 10
EPS = 1e-12


def normalize_tile_id(value):
    value = os.path.basename(str(value))
    value = re.sub(
        r"\.(tif|tiff|png|jpg|jpeg|npy|csv)$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"_(mask|cytox)$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value


def load_sytox():
    df = pd.read_csv(SYTOX_CSV)
    df["tile_id"] = df["filename"].map(normalize_tile_id)
    df["sytox"] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )
    df = df[np.isfinite(df["sytox"])].copy()

    return (
        df.groupby("tile_id")["sytox"]
        .mean()
        .to_dict()
    )


def load_latent(disease):
    X = np.load(
        LATENT_DIR / f"{disease}_latent_raw.npy",
        mmap_mode="r",
    )
    filenames = np.load(
        LATENT_DIR / f"{disease}_filenames.npy",
        allow_pickle=True,
    )

    if len(X) != len(filenames):
        raise ValueError(
            f"{disease}: latent vectors and filenames have different lengths."
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(
            [normalize_tile_id(x) for x in filenames],
            dtype=object,
        ),
    )


def make_overlapping_windows(n):
    if n < MIN_WINDOW_N:
        return []

    window_size = min(
        max(int(np.ceil(n * WINDOW_FRAC)), MIN_WINDOW_N),
        n,
    )
    max_start = n - window_size

    if max_start <= 0:
        starts = np.array([0], dtype=int)
    else:
        starts = np.rint(
            np.linspace(0, max_start, N_WINDOWS)
        ).astype(int)

    return [
        (i, int(start), int(start + window_size))
        for i, start in enumerate(starts, start=1)
    ]


def median_local_knn_radius(X, k=LOCAL_SCALE_K):
    X = np.asarray(X, dtype=np.float32)

    if len(X) <= k + 2:
        return np.nan

    knn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean",
        n_jobs=-1,
    )
    knn.fit(X)
    distances, _ = knn.kneighbors(X)

    radii = np.maximum(
        distances[:, k],
        EPS,
    )

    return float(np.median(radii))


def analyse_disease(disease, sytox_lookup):
    X, tile_ids = load_latent(disease)

    sytox = np.asarray(
        [sytox_lookup.get(x, np.nan) for x in tile_ids],
        dtype=float,
    )

    valid = np.isfinite(sytox) & (sytox > 0)
    X = X[valid]
    sytox = sytox[valid]

    order = np.argsort(sytox)
    X = X[order]
    sytox = sytox[order]

    rows = []

    for window_number, start, end in make_overlapping_windows(len(sytox)):
        X_window = X[start:end]
        sytox_window = sytox[start:end]

        rows.append(
            {
                "model": MODEL_NAME,
                "disease": disease,
                "window": window_number,
                "sytox_mean": float(np.mean(sytox_window)),
                "median_10nn_radius": median_local_knn_radius(
                    X_window
                ),
            }
        )

    return rows


def main():
    sytox_lookup = load_sytox()
    results = []

    for disease in DISEASES:
        results.extend(
            analyse_disease(
                disease,
                sytox_lookup,
            )
        )

    print(pd.DataFrame(results))


if __name__ == "__main__":
    main()
