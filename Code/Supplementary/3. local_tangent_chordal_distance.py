"""Local tangent chordal distance across SYTOX windows."""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


SYTOX_CSV = Path("path/to/sytox_cell_death.csv")
TARGET_COLUMN = "intensity_per_pixel_mean_3std"

MODEL_NAME = "SupMoCo"
LATENT_DIR = Path("path/to/supmoco_latent_vectors")

DISEASES = ["GBA", "PINK1", "SNCA"]

N_WINDOWS = 30
WINDOW_FRAC = 0.10
MIN_POINTS_PER_WINDOW = 100
MAX_POINTS_PER_WINDOW = 800

DENSITY_K = 10
DENSITY_TRIM_PERCENTILE = 95

GLOBAL_PCA_DIM = 50
PCA_FIT_MAX_POINTS = 10000

LOCAL_NEIGHBOR_K = 30
TANGENT_DIM = 10
N_COMPARE_NEIGHBORS = 5


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


def make_overlapping_windows(n):
    if n < MIN_POINTS_PER_WINDOW:
        return []

    window_size = max(
        int(np.floor(n * WINDOW_FRAC)),
        1,
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


def fit_global_pca(X):
    n_components = min(
        GLOBAL_PCA_DIM,
        X.shape[1],
        X.shape[0] - 1,
    )

    if len(X) > PCA_FIT_MAX_POINTS:
        rng = np.random.default_rng()
        indices = rng.choice(
            len(X),
            size=PCA_FIT_MAX_POINTS,
            replace=False,
        )
        X_fit = X[indices]
    else:
        X_fit = X

    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
    )
    pca.fit(X_fit)
    return pca


def subsample_window(X, target):
    if len(X) <= MAX_POINTS_PER_WINDOW:
        return X, target

    rng = np.random.default_rng()
    indices = rng.choice(
        len(X),
        size=MAX_POINTS_PER_WINDOW,
        replace=False,
    )
    return X[indices], target[indices]


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
    keep = np.isfinite(radii) & (radii <= threshold)

    return X[keep], target[keep]


def estimate_local_tangent_bases(X):
    X = np.asarray(X, dtype=np.float64)

    tangent_dim = min(
        TANGENT_DIM,
        LOCAL_NEIGHBOR_K - 1,
        X.shape[1],
    )

    knn = NearestNeighbors(
        n_neighbors=LOCAL_NEIGHBOR_K + 1,
        metric="euclidean",
        n_jobs=-1,
    )
    knn.fit(X)
    _, indices = knn.kneighbors(X)

    neighbor_indices = indices[:, 1:]

    bases = np.empty(
        (len(X), X.shape[1], tangent_dim),
        dtype=np.float32,
    )

    for i in range(len(X)):
        local_points = X[neighbor_indices[i]]
        local_centered = local_points - X[i]

        _, _, vt = np.linalg.svd(
            local_centered,
            full_matrices=False,
        )
        bases[i] = vt[:tangent_dim].T.astype(
            np.float32
        )

    return bases, neighbor_indices


def tangent_chordal_distance(U, V):
    angles = subspace_angles(
        np.asarray(U, dtype=np.float64),
        np.asarray(V, dtype=np.float64),
    )

    return float(
        np.sqrt(
            np.mean(np.sin(angles) ** 2)
        )
    )


def local_tangent_chordal_distance(X):
    bases, neighbor_indices = estimate_local_tangent_bases(X)

    n_compare = min(
        N_COMPARE_NEIGHBORS,
        LOCAL_NEIGHBOR_K,
    )

    point_chordal = np.full(
        len(X),
        np.nan,
        dtype=np.float64,
    )

    for i in range(len(X)):
        values = []

        for j in neighbor_indices[i, :n_compare]:
            values.append(
                tangent_chordal_distance(
                    bases[i],
                    bases[j],
                )
            )

        if values:
            point_chordal[i] = np.mean(values)

    return float(np.nanmedian(point_chordal))


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

    pca = fit_global_pca(X)
    X_geometry = pca.transform(X).astype(
        np.float32
    )

    rows = []

    for window_number, start, end in make_overlapping_windows(len(rates)):
        X_window = X_geometry[start:end]
        rates_window = rates[start:end]

        X_window, rates_window = subsample_window(
            X_window,
            rates_window,
        )
        X_window, rates_window = trim_upper_knn_radius(
            X_window,
            rates_window,
        )

        if (
            len(X_window) < MIN_POINTS_PER_WINDOW
            or len(X_window) <= LOCAL_NEIGHBOR_K + 2
        ):
            continue

        chordal_distance = local_tangent_chordal_distance(
            X_window
        )

        rows.append(
            {
                "model": MODEL_NAME,
                "disease": disease,
                "window": window_number,
                "sytox_mean": float(np.mean(rates_window)),
                "local_tangent_chordal_distance": chordal_distance,
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
