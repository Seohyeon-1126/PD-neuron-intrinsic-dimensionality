"""GRIDE intrinsic dimensionality across SYTOX windows."""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import betaln
from sklearn.neighbors import NearestNeighbors


SYTOX_CSV = Path("path/to/sytox_cell_death.csv")
TARGET_COLUMN = "intensity_per_pixel_mean_3std"

MODEL_NAME = "SupMoCo"
LATENT_DIR = Path("path/to/supmoco_latent_vectors")

DISEASES = ["GBA", "PINK1", "SNCA"]

N_WINDOWS = 30
WINDOW_FRAC = 0.10
MIN_POINTS = 50

DENSITY_K = 10
DENSITY_TRIM_PERCENTILE = 95

GRIDE_N1 = 10
GRIDE_N2 = 20
MAX_POINTS = 2500

MIN_DIMENSION = 0.10
MAX_DIMENSION_MULTIPLIER = 4.0
OPTIMIZER_XATOL = 1e-6


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
    if n < MIN_POINTS:
        return []

    window_size = min(
        max(int(np.ceil(n * WINDOW_FRAC)), MIN_POINTS),
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

    if keep.sum() < MIN_POINTS:
        return X, target

    return X[keep], target[keep]


def subsample_rows(X):
    if len(X) <= MAX_POINTS:
        return X

    rng = np.random.default_rng()
    indices = rng.choice(
        len(X),
        size=MAX_POINTS,
        replace=False,
    )
    return X[indices]


def log_expm1_positive(values):
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)

    small = values < 50
    output[small] = np.log(np.expm1(values[small]))
    output[~small] = values[~small] + np.log1p(
        -np.exp(-values[~small])
    )
    return output


def gride_log_likelihood(dimension, log_ratio, n1, n2):
    if not np.isfinite(dimension) or dimension <= 0:
        return -np.inf

    exponent = dimension * log_ratio
    if np.any(exponent <= 0) or not np.all(np.isfinite(exponent)):
        return -np.inf

    n = len(log_ratio)
    gap = n2 - n1

    value = n * np.log(dimension)
    value -= n * betaln(gap, n1)
    value -= ((n2 - 1) * dimension + 1) * np.sum(log_ratio)

    if gap > 1:
        value += (gap - 1) * np.sum(
            log_expm1_positive(exponent)
        )

    return float(value)


def gride_intrinsic_dimension(
    X,
    n1=GRIDE_N1,
    n2=GRIDE_N2,
    eps=1e-12,
):
    X = subsample_rows(
        np.asarray(X, dtype=np.float32)
    )

    minimum_required = max(MIN_POINTS, n2 + 2)
    if len(X) < minimum_required:
        return np.nan

    knn = NearestNeighbors(
        n_neighbors=n2 + 1,
        metric="euclidean",
        n_jobs=-1,
    )
    knn.fit(X)
    distances, _ = knn.kneighbors(X)

    r_n1 = distances[:, n1]
    r_n2 = distances[:, n2]

    valid = (
        np.isfinite(r_n1)
        & np.isfinite(r_n2)
        & (r_n1 > eps)
        & (r_n2 > r_n1)
    )

    ratios = r_n2[valid] / r_n1[valid]
    ratios = ratios[
        np.isfinite(ratios)
        & (ratios > 1 + eps)
    ]

    if len(ratios) < minimum_required:
        return np.nan

    log_ratio = np.log(ratios)
    max_dimension = max(
        100.0,
        X.shape[1] * MAX_DIMENSION_MULTIPLIER,
    )

    def objective(log_dimension):
        dimension = float(np.exp(log_dimension))
        log_likelihood = gride_log_likelihood(
            dimension,
            log_ratio,
            n1,
            n2,
        )
        return (
            -log_likelihood
            if np.isfinite(log_likelihood)
            else np.inf
        )

    result = minimize_scalar(
        objective,
        bounds=(
            np.log(MIN_DIMENSION),
            np.log(max_dimension),
        ),
        method="bounded",
        options={"xatol": OPTIMIZER_XATOL},
    )

    return (
        float(np.exp(result.x))
        if result.success
        else np.nan
    )


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

        gride_id = gride_intrinsic_dimension(X_trimmed)

        rows.append(
            {
                "model": MODEL_NAME,
                "disease": disease,
                "window": window_number,
                "sytox_mean": float(np.mean(rates_trimmed)),
                "gride_id": gride_id,
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
