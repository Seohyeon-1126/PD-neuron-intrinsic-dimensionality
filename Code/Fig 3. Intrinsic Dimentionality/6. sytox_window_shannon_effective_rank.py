"""Shannon effective rank across SYTOX windows."""

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

DENSITY_K = 10
DENSITY_TRIM_PERCENTILE = 95
MAX_POINTS = 2500

EIGENVALUE_RELATIVE_EPS = 1e-12


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


def trim_upper_knn_radius(X, target):
    if len(X) <= DENSITY_K + 2:
        return X, target

    knn = NearestNeighbors(
        n_neighbors=DENSITY_K + 1,
        metric="euclidean",
        n_jobs=-1,
    )
    knn.fit(X)
    distances, _ = knn.kneighbors(X)

    radii = distances[:, DENSITY_K]
    threshold = np.percentile(
        radii,
        DENSITY_TRIM_PERCENTILE,
    )
    keep = radii <= threshold

    if keep.sum() < MIN_WINDOW_N:
        return X, target

    return X[keep], target[keep]


def cap_points(X):
    if len(X) <= MAX_POINTS:
        return X

    rng = np.random.default_rng()
    indices = rng.choice(
        len(X),
        size=MAX_POINTS,
        replace=False,
    )
    return X[indices]


def shannon_effective_rank(X):
    X = np.asarray(X, dtype=np.float64)

    if len(X) < MIN_WINDOW_N:
        return np.nan, np.nan

    X = cap_points(X)
    X = X - np.mean(X, axis=0, keepdims=True)

    singular_values = np.linalg.svd(
        X,
        full_matrices=False,
        compute_uv=False,
    )

    eigenvalues = (
        singular_values ** 2
        / max(len(X) - 1, 1)
    )
    eigenvalues = eigenvalues[
        np.isfinite(eigenvalues)
        & (eigenvalues > 0)
    ]

    if len(eigenvalues) == 0:
        return np.nan, np.nan

    threshold = (
        np.max(eigenvalues)
        * EIGENVALUE_RELATIVE_EPS
    )
    eigenvalues = eigenvalues[
        eigenvalues > threshold
    ]

    probabilities = eigenvalues / np.sum(eigenvalues)
    probabilities = probabilities[
        np.isfinite(probabilities)
        & (probabilities > 0)
    ]
    probabilities = probabilities / np.sum(probabilities)

    entropy = float(
        -np.sum(
            probabilities
            * np.log(probabilities)
        )
    )
    effective_rank = float(np.exp(entropy))

    return entropy, effective_rank


def load_sytox():
    df = pd.read_csv(SYTOX_CSV)
    df["tile_id"] = df["filename"].map(normalize_tile_id)
    df["sytox_rate"] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )
    df = df[np.isfinite(df["sytox_rate"])].copy()

    return (
        df.groupby("tile_id")["sytox_rate"]
        .mean()
        .to_dict()
    )


def load_latent(disease):
    latent_path = LATENT_DIR / f"{disease}_latent_raw.npy"
    filename_path = LATENT_DIR / f"{disease}_filenames.npy"

    X = np.load(latent_path)
    filenames = np.load(
        filename_path,
        allow_pickle=True,
    )

    if len(X) != len(filenames):
        raise ValueError(
            f"{disease}: latent vectors and filenames have different lengths."
        )

    return np.asarray(X, dtype=np.float32), filenames


def analyse_disease(disease, sytox_lookup):
    X, filenames = load_latent(disease)

    tile_ids = np.array(
        [normalize_tile_id(x) for x in filenames],
        dtype=object,
    )
    rates = np.array(
        [sytox_lookup.get(x, np.nan) for x in tile_ids],
        dtype=float,
    )

    valid = np.isfinite(rates) & (rates > 0)
    X = X[valid]
    rates = rates[valid]

    order = np.argsort(rates)
    X = X[order]
    rates = rates[order]

    rows = []

    for window_number, start, end in make_overlapping_windows(len(rates)):
        X_window = X[start:end]
        rates_window = rates[start:end]

        X_trimmed, rates_trimmed = trim_upper_knn_radius(
            X_window,
            rates_window,
        )

        entropy, effective_rank = shannon_effective_rank(
            X_trimmed
        )

        rows.append(
            {
                "model": MODEL_NAME,
                "disease": disease,
                "window": window_number,
                "sytox_mean": float(np.mean(rates_trimmed)),
                "shannon_entropy": entropy,
                "shannon_effective_rank": effective_rank,
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

    results_df = pd.DataFrame(results)
    print(results_df)


if __name__ == "__main__":
    main()
