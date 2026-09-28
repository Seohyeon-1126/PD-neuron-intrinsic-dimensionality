"""TWO-NN sensitivity analysis across SYTOX windows."""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


DATA_DIR = Path("data")
LATENT_DIR = Path("latent_vectors")

SYTOX_CSV = Path("path/to/sytox_cell_death.csv")
TARGET_COLUMN = "intensity_per_pixel_mean_3std"

MODEL_NAME = "SupMoCo"
MODEL_DIR = Path("path/to/supmoco_latent_vectors")

DISEASES = ["GBA", "PINK1", "SNCA"]

N_WINDOWS = 30
WINDOW_FRAC = 0.10
MIN_WINDOW_N = 50

DENSITY_K = 10
DENSITY_TRIM_PERCENTILE = 95
MAX_POINTS = 2500

ESTIMATORS = ["linear_fit", "pareto_mle"]
MU_TRIM_PERCENTILES = [None, 98, 95]
DECIMATION_FRACTIONS = [1.00, 0.70, 0.50, 0.30]
N_DECIMATION_REPEATS = 20

MIN_TWONN_POINTS = 50
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

    nn = NearestNeighbors(
        n_neighbors=DENSITY_K + 1,
        metric="euclidean",
        n_jobs=-1,
    )
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    radii = distances[:, DENSITY_K]
    threshold = np.percentile(
        radii,
        DENSITY_TRIM_PERCENTILE,
    )
    keep = radii <= threshold

    if keep.sum() < MIN_WINDOW_N:
        return X, target

    return X[keep], target[keep]


def decimate_rows(X, fraction):
    if fraction >= 1.0:
        return np.asarray(X, dtype=np.float32)

    n_keep = max(
        MIN_TWONN_POINTS,
        int(np.floor(len(X) * fraction)),
    )
    n_keep = min(n_keep, len(X))

    rng = np.random.default_rng()
    idx = rng.choice(
        len(X),
        size=n_keep,
        replace=False,
    )
    return np.asarray(X[idx], dtype=np.float32)


def cap_points(X):
    if len(X) <= MAX_POINTS:
        return X

    rng = np.random.default_rng()
    idx = rng.choice(
        len(X),
        size=MAX_POINTS,
        replace=False,
    )
    return X[idx]


def get_mu_ratios(X):
    X = np.asarray(X, dtype=np.float32)

    if len(X) < MIN_TWONN_POINTS:
        return np.array([], dtype=float)

    nn = NearestNeighbors(
        n_neighbors=3,
        metric="euclidean",
        n_jobs=-1,
    )
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    r1 = distances[:, 1]
    r2 = distances[:, 2]

    valid = (
        np.isfinite(r1)
        & np.isfinite(r2)
        & (r1 > EPS)
        & (r2 > r1)
    )

    mu = r2[valid] / r1[valid]
    return mu[np.isfinite(mu) & (mu > 1.0 + EPS)]


def trim_mu(mu, percentile):
    if percentile is None:
        return mu

    threshold = np.percentile(mu, percentile)
    return mu[mu <= threshold]


def twonn_linear_fit(mu):
    if len(mu) < MIN_TWONN_POINTS:
        return np.nan

    mu = np.sort(mu)
    n = len(mu)

    empirical_cdf = (
        np.arange(1, n + 1, dtype=np.float64)
        / (n + 1)
    )

    x = np.log(mu)
    y = -np.log(1.0 - empirical_cdf)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x > EPS)
        & (y > 0)
    )
    x = x[valid]
    y = y[valid]

    if len(x) < MIN_TWONN_POINTS:
        return np.nan

    denominator = np.sum(x ** 2)
    if denominator <= EPS:
        return np.nan

    return float(np.sum(x * y) / denominator)


def twonn_pareto_mle(mu):
    if len(mu) < MIN_TWONN_POINTS:
        return np.nan

    denominator = np.sum(np.log(mu))

    if not np.isfinite(denominator) or denominator <= EPS:
        return np.nan

    return float((len(mu) - 1) / denominator)


def estimate_twonn(
    X,
    estimator,
    mu_trim_percentile,
    fraction,
):
    X = decimate_rows(X, fraction)
    X = cap_points(X)

    mu = get_mu_ratios(X)
    if len(mu) < MIN_TWONN_POINTS:
        return np.nan

    mu = trim_mu(mu, mu_trim_percentile)

    if estimator == "linear_fit":
        return twonn_linear_fit(mu)
    if estimator == "pareto_mle":
        return twonn_pareto_mle(mu)

    raise ValueError(f"Unknown estimator: {estimator}")


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
    latent_path = MODEL_DIR / f"{disease}_latent_raw.npy"
    filename_path = MODEL_DIR / f"{disease}_filenames.npy"

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

        for estimator in ESTIMATORS:
            for mu_trim_percentile in MU_TRIM_PERCENTILES:
                for fraction in DECIMATION_FRACTIONS:
                    repeat_values = []

                    for repeat in range(N_DECIMATION_REPEATS):
                        repeat_values.append(
                            estimate_twonn(
                                X_trimmed,
                                estimator,
                                mu_trim_percentile,
                                fraction,
                            )
                        )

                    repeat_values = np.asarray(
                        repeat_values,
                        dtype=float,
                    )
                    valid_values = repeat_values[
                        np.isfinite(repeat_values)
                    ]

                    trim_label = (
                        "none"
                        if mu_trim_percentile is None
                        else f"upper_{100 - mu_trim_percentile:g}pct_removed"
                    )

                    rows.append(
                        {
                            "model": MODEL_NAME,
                            "disease": disease,
                            "window": window_number,
                            "sytox_mean": float(
                                np.mean(rates_trimmed)
                            ),
                            "estimator": estimator,
                            "mu_trim": trim_label,
                            "decimation_fraction": fraction,
                            "id_mean": (
                                float(np.mean(valid_values))
                                if len(valid_values)
                                else np.nan
                            ),
                            "id_sd": (
                                float(np.std(valid_values, ddof=1))
                                if len(valid_values) > 1
                                else 0.0
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

    results_df = pd.DataFrame(results)
    print(results_df)


if __name__ == "__main__":
    main()
