"""SYTOX-windowed intrinsic dimensionality analysis using corrected MLE."""

from pathlib import Path
import os
import re

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


DATA_DIR = Path("data")
LATENT_DIR = Path("latent_vectors")

SYTOX_CSV = Path("path/to/sytox_cell_death.csv")
TARGET_COLUMN = "intensity_per_pixel_mean_3std"

MODEL_PATHS = {
    "ResNet18": Path("path/to/resnet18_latent_vectors"),
    "SupMoCo": Path("path/to/supmoco_latent_vectors"),
}

DISEASES = ["GBA", "PINK1", "SNCA"]

N_WINDOWS = 30
WINDOW_FRAC = 0.10
MIN_WINDOW_N = 50

MLE_K = 20
MAX_POINTS = 2500

DENSITY_K = 10
UPPER_RADIUS_PERCENTILE = 95



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


def get_knn_radius(X, k=DENSITY_K, eps=1e-12):
    if len(X) <= k + 2:
        return None

    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean",
        n_jobs=-1,
    )
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    return np.maximum(distances[:, k], eps)


def trim_upper_knn_radius(X, rates):
    radii = get_knn_radius(X)

    if radii is None:
        return X, rates

    threshold = np.percentile(
        radii,
        UPPER_RADIUS_PERCENTILE,
    )
    keep = radii <= threshold

    if keep.sum() < MIN_WINDOW_N:
        return X, rates

    return X[keep], rates[keep]


def subsample_rows(X, max_points=MAX_POINTS):
    if len(X) <= max_points:
        return X

    rng = np.random.default_rng()
    idx = rng.choice(
        len(X),
        size=max_points,
        replace=False,
    )
    return X[idx]


def corrected_mle_id(
    X,
    k=MLE_K,
    max_points=MAX_POINTS,
    eps=1e-12,
):
    """Finite-sample-corrected Levina-Bickel MLE."""
    X = np.asarray(X, dtype=np.float32)

    if len(X) <= k + 2:
        return np.nan

    X = subsample_rows(
        X,
        max_points=max_points,
    )

    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean",
        n_jobs=-1,
    )
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    distances = np.maximum(
        distances[:, 1:],
        eps,
    )

    T_k = distances[:, k - 1]
    T_j = distances[:, : k - 1]

    sum_logs = np.sum(
        np.log(T_k[:, None] / T_j),
        axis=1,
    )

    valid = np.isfinite(sum_logs) & (sum_logs > eps)

    if valid.sum() < 10:
        return np.nan

    local_id = (k - 2) / sum_logs[valid]
    return float(np.mean(local_id))


def make_overlapping_windows(
    n,
    n_windows=N_WINDOWS,
    frac=WINDOW_FRAC,
):
    """Create overlapping windows after sorting samples by SYTOX burden."""
    if n < MIN_WINDOW_N:
        return []

    window_size = max(
        int(np.ceil(n * frac)),
        MIN_WINDOW_N,
    )
    window_size = min(window_size, n)

    max_start = n - window_size

    if max_start <= 0:
        starts = np.array([0], dtype=int)
    else:
        starts = np.rint(
            np.linspace(0, max_start, n_windows)
        ).astype(int)

    return [
        (i, int(start), int(start + window_size))
        for i, start in enumerate(starts, start=1)
    ]


def load_sytox():
    df = pd.read_csv(SYTOX_CSV)

    df["tile_id"] = df["filename"].map(normalize_tile_id)
    df["sytox_rate"] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    df = df[
        np.isfinite(df["sytox_rate"])
    ].copy()

    return (
        df.groupby("tile_id")["sytox_rate"]
        .mean()
        .to_dict()
    )


def load_latent(model_dir, disease):
    latent_path = model_dir / f"{disease}_latent_raw.npy"
    filename_path = model_dir / f"{disease}_filenames.npy"

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


def analyse_model_disease(
    model_name,
    model_dir,
    disease,
    sytox_lookup,
):
    X, filenames = load_latent(model_dir, disease)

    tile_ids = np.array(
        [normalize_tile_id(x) for x in filenames],
        dtype=object,
    )
    rates = np.array(
        [sytox_lookup.get(x, np.nan) for x in tile_ids],
        dtype=float,
    )

    # Use matched samples with positive SYTOX values.
    valid = np.isfinite(rates) & (rates > 0)
    X = X[valid]
    rates = rates[valid]

    if len(rates) < MIN_WINDOW_N:
        raise ValueError(
            f"{model_name}/{disease}: too few matched samples."
        )

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

        mle_id = corrected_mle_id(
            X_trimmed,
        )

        rows.append(
            {
                "model": model_name,
                "disease": disease,
                "window": window_number,
                "sytox_mean": float(np.mean(rates_trimmed)),
                "sytox_median": float(np.median(rates_trimmed)),
                "mle_id": mle_id,
                "n_before_trim": len(rates_window),
                "n_after_trim": len(rates_trimmed),
            }
        )

    return rows


def main():
    sytox_lookup = load_sytox()
    results = []

    for model_name, model_dir in MODEL_PATHS.items():
        for disease in DISEASES:
            results.extend(
                analyse_model_disease(
                    model_name,
                    model_dir,
                    disease,
                    sytox_lookup,
                )
            )

    results_df = pd.DataFrame(results)
    print(results_df)


if __name__ == "__main__":
    main()
