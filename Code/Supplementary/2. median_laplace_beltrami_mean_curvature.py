"""Median Laplace-Beltrami mean-curvature magnitude across SYTOX windows."""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


SYTOX_CSV = Path("path/to/sytox_cell_death.csv")
TARGET_COLUMN = "intensity_per_pixel_mean_3std"

MODEL_NAME = "SupMoCo"
LATENT_DIR = Path("path/to/supmoco_latent_vectors")

DISEASES = ["GBA", "PINK1", "SNCA"]

GRAPH_K = 30
BANDWIDTH_K = 20
ID_K = 20

EPSILON = 1.0
COIFMAN_ALPHA = 1.0
MIN_LOCAL_ID = 2.0
MAX_LOCAL_ID = 100.0
EPS = 1e-12

N_WINDOWS = 30
WINDOW_FRAC = 0.10
CURVATURE_TRIM_UPPER = 99.5


def normalize_tile_id(value):
    value = os.path.basename(str(value))
    value = re.sub(
        r"\.(tif|tiff|png|jpg|jpeg|npy|csv)$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"_(mask|sytox|cytox)$",
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
    df = df.dropna(subset=["tile_id", "sytox"])
    df = df.groupby("tile_id", as_index=False)["sytox"].mean()
    return dict(zip(df["tile_id"], df["sytox"]))


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

    X = np.asarray(X, dtype=np.float32)
    tile_ids = np.asarray(
        [normalize_tile_id(x) for x in filenames],
        dtype=object,
    )
    return X, tile_ids


def compute_knn(X, n_neighbors):
    try:
        import faiss

        Xc = np.ascontiguousarray(X, dtype=np.float32)
        index = faiss.IndexFlatL2(Xc.shape[1])
        index.add(Xc)

        squared_distances, indices = index.search(
            Xc,
            min(n_neighbors + 1, len(Xc)),
        )

        all_distances = np.empty(
            (len(Xc), n_neighbors),
            dtype=np.float32,
        )
        all_indices = np.empty(
            (len(Xc), n_neighbors),
            dtype=np.int64,
        )

        for i in range(len(Xc)):
            keep = indices[i] != i
            idx = indices[i][keep][:n_neighbors]
            d2 = squared_distances[i][keep][:n_neighbors]

            if len(idx) < n_neighbors:
                raise RuntimeError("Insufficient nearest neighbours.")

            all_indices[i] = idx
            all_distances[i] = np.sqrt(np.maximum(d2, 0.0))

        return all_distances, all_indices

    except Exception as error:
        warnings.warn(f"FAISS unavailable; using sklearn: {error}")

        nn = NearestNeighbors(
            n_neighbors=n_neighbors + 1,
            metric="euclidean",
            n_jobs=-1,
        )
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        return (
            distances[:, 1:].astype(np.float32),
            indices[:, 1:].astype(np.int64),
        )


def compute_local_bandwidths(knn_distances):
    sigma = np.median(
        knn_distances[:, :BANDWIDTH_K],
        axis=1,
    ).astype(np.float64)

    positive = sigma[np.isfinite(sigma) & (sigma > EPS)]
    if len(positive) == 0:
        raise RuntimeError("No valid adaptive bandwidths.")

    replacement = float(np.median(positive))
    sigma[~np.isfinite(sigma) | (sigma <= EPS)] = replacement
    return sigma


def estimate_local_id(knn_distances):
    distances = np.asarray(
        knn_distances[:, :ID_K],
        dtype=np.float64,
    )
    distances = np.maximum(distances, EPS)

    r_k = distances[:, ID_K - 1]
    logs = np.log(
        r_k[:, None] / distances[:, :ID_K - 1]
    )
    denominator = np.mean(logs, axis=1)

    local_id = np.divide(
        1.0,
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > EPS,
    )

    valid = np.isfinite(local_id) & (local_id > 0)
    if not np.any(valid):
        raise RuntimeError("No valid local intrinsic-dimension estimates.")

    local_id[~valid] = np.nanmedian(local_id[valid])
    return np.clip(
        local_id,
        MIN_LOCAL_ID,
        MAX_LOCAL_ID,
    )


def build_adaptive_kernel(knn_distances, knn_indices, sigma):
    n, k = knn_indices.shape

    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = knn_indices.reshape(-1)
    distances = knn_distances.reshape(-1).astype(np.float64)

    denominator = (
        EPSILON
        * sigma[rows]
        * sigma[cols]
    )
    denominator = np.maximum(denominator, EPS)

    weights = np.exp(-(distances ** 2) / denominator)
    weights[~np.isfinite(weights)] = 0.0

    K = sparse.csr_matrix(
        (weights, (rows, cols)),
        shape=(n, n),
        dtype=np.float64,
    )

    K = (K + K.T) * 0.5
    K.setdiag(0.0)
    K.eliminate_zeros()

    return K.tocsr()


def coifman_lafon_normalize(K):
    q = np.asarray(K.sum(axis=1)).ravel()

    positive = q[np.isfinite(q) & (q > EPS)]
    if len(positive) == 0:
        raise RuntimeError("Kernel degree is zero.")

    q[~np.isfinite(q) | (q <= EPS)] = np.median(positive)

    Q_inv = sparse.diags(
        np.power(q, -COIFMAN_ALPHA),
        format="csr",
    )
    K_alpha = (Q_inv @ K @ Q_inv).tocsr()

    degree = np.asarray(K_alpha.sum(axis=1)).ravel()
    positive = degree[np.isfinite(degree) & (degree > EPS)]
    if len(positive) == 0:
        raise RuntimeError("Density-corrected degree is zero.")

    degree[~np.isfinite(degree) | (degree <= EPS)] = np.median(positive)

    D_inv = sparse.diags(
        1.0 / degree,
        format="csr",
    )
    return (D_inv @ K_alpha).tocsr()


def estimate_lb_mean_curvature(X, P, sigma, local_id):
    X64 = np.asarray(X, dtype=np.float64)

    displacement = P.dot(X64) - X64
    local_scale_squared = np.maximum(
        EPSILON * np.square(sigma),
        EPS,
    )

    laplace_vector = (
        displacement
        / local_scale_squared[:, None]
    )
    laplace_vector_norm = np.linalg.norm(
        laplace_vector,
        axis=1,
    )

    safe_id = np.maximum(local_id, 1.0)
    mean_curvature = laplace_vector_norm / safe_id
    mean_curvature_scale_norm = mean_curvature * sigma

    return (
        mean_curvature,
        mean_curvature_scale_norm,
        laplace_vector_norm,
    )


def curvature_valid_mask(
    mean_curvature,
    mean_curvature_scale_norm,
    laplace_vector_norm,
):
    valid = np.ones(len(mean_curvature), dtype=bool)

    for values in (
        mean_curvature,
        mean_curvature_scale_norm,
        laplace_vector_norm,
    ):
        values = np.asarray(values, dtype=np.float64)
        metric_valid = np.isfinite(values)

        if np.any(metric_valid):
            upper = np.nanpercentile(
                values[metric_valid],
                CURVATURE_TRIM_UPPER,
            )
            metric_valid &= values <= upper

        valid &= metric_valid

    return valid


def make_sytox_windows(df):
    df = df[
        df["sytox"].notna()
        & (df["sytox"] > 0)
        & df["curvature_valid"]
    ].copy()

    df = df.sort_values("sytox").reset_index(drop=True)
    n = len(df)

    if n == 0:
        return pd.DataFrame()

    window_size = max(
        int(np.floor(n * WINDOW_FRAC)),
        1,
    )

    if n <= window_size:
        starts = np.asarray([0], dtype=int)
    else:
        starts = np.rint(
            np.linspace(
                0,
                n - window_size,
                N_WINDOWS,
            )
        ).astype(int)

    rows = []

    for window_number, start in enumerate(starts, start=1):
        subset = df.iloc[start:start + window_size]

        rows.append(
            {
                "window": window_number,
                "mean_sytox": float(subset["sytox"].mean()),
                "median_lb_mean_curvature_magnitude": float(
                    subset["mean_curvature"].median()
                ),
            }
        )

    return pd.DataFrame(rows)


def analyse_disease(disease, sytox_lookup):
    X, tile_ids = load_latent(disease)

    max_k = max(
        GRAPH_K,
        BANDWIDTH_K,
        ID_K,
    )
    knn_distances, knn_indices = compute_knn(
        X,
        max_k,
    )

    sigma = compute_local_bandwidths(knn_distances)
    local_id = estimate_local_id(knn_distances)

    K = build_adaptive_kernel(
        knn_distances[:, :GRAPH_K],
        knn_indices[:, :GRAPH_K],
        sigma,
    )
    P = coifman_lafon_normalize(K)

    (
        mean_curvature,
        mean_curvature_scale_norm,
        laplace_vector_norm,
    ) = estimate_lb_mean_curvature(
        X,
        P,
        sigma,
        local_id,
    )

    point_df = pd.DataFrame(
        {
            "tile_id": tile_ids,
            "mean_curvature": mean_curvature,
            "curvature_valid": curvature_valid_mask(
                mean_curvature,
                mean_curvature_scale_norm,
                laplace_vector_norm,
            ),
        }
    )

    point_df["sytox"] = point_df["tile_id"].map(
        sytox_lookup
    )

    window_df = make_sytox_windows(point_df)

    if not window_df.empty:
        window_df.insert(0, "disease", disease)
        window_df.insert(0, "model", MODEL_NAME)

    return window_df


def main():
    sytox_lookup = load_sytox()

    results = [
        analyse_disease(disease, sytox_lookup)
        for disease in DISEASES
    ]
    results = [df for df in results if not df.empty]

    if results:
        print(
            pd.concat(
                results,
                ignore_index=True,
            )
        )


if __name__ == "__main__":
    main()
