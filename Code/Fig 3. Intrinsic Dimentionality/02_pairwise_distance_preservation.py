"""Seed-to-seed preservation of pairwise latent distances."""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


LATENT_DIR = Path("latent_vectors")

MODEL_PATHS = {
    "ResNet18_seed42": LATENT_DIR / "ResNet18_seed42",
    "ResNet18_seed123": LATENT_DIR / "ResNet18_seed123",
    "SupMoCo_seed42": LATENT_DIR / "SupMoCo_seed42",
    "SupMoCo_seed123": LATENT_DIR / "SupMoCo_seed123",
    "SupMoCo_seed2024": LATENT_DIR / "SupMoCo_seed2024",
}

MODEL_PATHS = {
    "ResNet18": Path("path/to/resnet18_latent_vectors"),
    "SupMoCo": Path("path/to/supmoco_latent_vectors"),
}

DISEASES = ["GBA", "PINK1", "SNCA"]

MAX_COMMON_TILES = 10_000
N_RANDOM_PAIRS = 100_000
RANDOM_SEED = 42


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


def load_raw_latent(model_name, disease):
    latent_dir = MODEL_PATHS[model_name]

    X = np.load(
        latent_dir / f"{disease}_latent_raw.npy",
        mmap_mode="r",
    )
    filenames = np.load(
        latent_dir / f"{disease}_filenames.npy",
        allow_pickle=True,
    )

    if len(X) != len(filenames):
        raise ValueError(
            f"{model_name}/{disease}: latent vectors and filenames have different lengths."
        )

    X = np.asarray(X, dtype=np.float32)
    tile_ids = np.asarray(
        [normalize_tile_id(x) for x in filenames],
        dtype=object,
    )

    return X, tile_ids


def match_common_tiles(X1, ids1, X2, ids2, max_tiles, seed):
    index1 = {tile_id: i for i, tile_id in enumerate(ids1)}
    index2 = {tile_id: i for i, tile_id in enumerate(ids2)}

    common_ids = sorted(set(index1) & set(index2))

    if len(common_ids) < 100:
        raise ValueError("Too few common tiles.")

    if len(common_ids) > max_tiles:
        rng = np.random.default_rng(seed)
        selected = np.sort(
            rng.choice(
                len(common_ids),
                size=max_tiles,
                replace=False,
            )
        )
        common_ids = [common_ids[i] for i in selected]

    idx1 = np.asarray([index1[x] for x in common_ids], dtype=int)
    idx2 = np.asarray([index2[x] for x in common_ids], dtype=int)

    return X1[idx1], X2[idx2], len(common_ids)


def generate_random_pairs(n_points, n_pairs, seed):
    rng = np.random.default_rng(seed)

    pair_i = rng.integers(0, n_points, size=n_pairs)
    pair_j = rng.integers(0, n_points, size=n_pairs)

    same = pair_i == pair_j
    while np.any(same):
        pair_j[same] = rng.integers(
            0,
            n_points,
            size=int(np.sum(same)),
        )
        same = pair_i == pair_j

    return pair_i, pair_j


def pairwise_distances(X, pair_i, pair_j):
    return np.linalg.norm(
        X[pair_i] - X[pair_j],
        axis=1,
    ).astype(np.float64)


def compare_pairwise_geometry(X1, X2, n_pairs, seed):
    pair_i, pair_j = generate_random_pairs(
        len(X1),
        n_pairs,
        seed,
    )

    d1 = pairwise_distances(X1, pair_i, pair_j)
    d2 = pairwise_distances(X2, pair_i, pair_j)

    valid = (
        np.isfinite(d1)
        & np.isfinite(d2)
        & (d1 > 0)
        & (d2 > 0)
    )

    d1 = d1[valid]
    d2 = d2[valid]

    pearson_r = pearsonr(d1, d2).statistic
    spearman_rho = spearmanr(d1, d2).statistic

    d1_scaled = d1 / np.median(d1)
    d2_scaled = d2 / np.median(d2)

    normalized_stress = np.sqrt(
        np.sum((d1_scaled - d2_scaled) ** 2)
        / np.sum(d1_scaled ** 2)
    )

    log_distance_ratio_sd = np.std(
        np.log(d2 / d1)
    )

    return {
        "pearson_r": float(pearson_r),
        "spearman_rho": float(spearman_rho),
        "normalized_stress": float(normalized_stress),
        "log_distance_ratio_sd": float(log_distance_ratio_sd),
        "n_pairs_valid": int(len(d1)),
    }


def main():
    rows = []

    for pair_index, (family, model_1, model_2, seed_pair) in enumerate(
        MODEL_PAIRS,
        start=1,
    ):
        for disease_index, disease in enumerate(DISEASES, start=1):
            X1, ids1 = load_raw_latent(model_1, disease)
            X2, ids2 = load_raw_latent(model_2, disease)

            if X1.shape[1] != X2.shape[1]:
                raise ValueError(
                    f"{model_1}/{model_2}/{disease}: latent dimensions differ."
                )

            X1, X2, n_common = match_common_tiles(
                X1,
                ids1,
                X2,
                ids2,
                max_tiles=MAX_COMMON_TILES,
                seed=RANDOM_SEED + pair_index * 100 + disease_index,
            )

            metrics = compare_pairwise_geometry(
                X1,
                X2,
                n_pairs=N_RANDOM_PAIRS,
                seed=RANDOM_SEED + pair_index * 10_000 + disease_index * 1_000,
            )

            rows.append(
                {
                    "model_family": family,
                    "seed_pair": seed_pair,
                    "disease": disease,
                    "n_common_tiles": n_common,
                    **metrics,
                }
            )

    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
