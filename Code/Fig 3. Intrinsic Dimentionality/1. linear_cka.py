"""Linear CKA between latent representations."""

import os
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_DIRS = {
    "model_1": Path("path/to/model_1_latent_vectors"),
    "model_2": Path("path/to/model_2_latent_vectors"),
}

GROUPS = ["GBA", "PINK1", "SNCA"]
MAX_MATCHED_TILES = 10000


def normalize_filename(value):
    value = os.path.basename(str(value))
    return re.sub(
        r"\.(tif|tiff|png|jpg|jpeg|npy)$",
        "",
        value,
        flags=re.IGNORECASE,
    )


def load_latent(model_name, group):
    latent_dir = MODEL_DIRS[model_name]

    X = np.load(
        latent_dir / f"{group}_latent_raw.npy",
        mmap_mode="r",
    )
    filenames = np.load(
        latent_dir / f"{group}_filenames.npy",
        allow_pickle=True,
    )

    X = np.asarray(X, dtype=np.float32)
    filenames = np.asarray(
        [normalize_filename(x) for x in filenames],
        dtype=object,
    )

    if len(X) != len(filenames):
        raise ValueError(
            f"{model_name}/{group}: latent vectors and filenames have different lengths."
        )

    return X, filenames


def match_identical_tiles(X, names_x, Y, names_y):
    index_x = {name: i for i, name in enumerate(names_x)}
    index_y = {name: i for i, name in enumerate(names_y)}

    common_names = sorted(
        set(index_x) & set(index_y)
    )

    if not common_names:
        raise ValueError("No identical image tiles were found.")

    idx_x = np.asarray(
        [index_x[name] for name in common_names],
        dtype=int,
    )
    idx_y = np.asarray(
        [index_y[name] for name in common_names],
        dtype=int,
    )

    return (
        X[idx_x],
        Y[idx_y],
        np.asarray(common_names, dtype=object),
    )


def subsample_matched(X, Y, names):
    if len(names) <= MAX_MATCHED_TILES:
        return X, Y, names

    rng = np.random.default_rng()
    idx = rng.choice(
        len(names),
        size=MAX_MATCHED_TILES,
        replace=False,
    )

    return X[idx], Y[idx], names[idx]


def center_features(X):
    X = np.asarray(X, dtype=np.float64)
    return X - np.mean(X, axis=0, keepdims=True)


def linear_cka(X, Y):
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            "X and Y must contain the same observations."
        )

    X = center_features(X)
    Y = center_features(Y)

    cross = X.T @ Y
    self_x = X.T @ X
    self_y = Y.T @ Y

    numerator = np.sum(cross ** 2)
    denominator = (
        np.sqrt(np.sum(self_x ** 2))
        * np.sqrt(np.sum(self_y ** 2))
    )

    if not np.isfinite(denominator) or denominator <= 0:
        return np.nan

    return float(numerator / denominator)


def main():
    rows = []

    for group in GROUPS:
        loaded = {
            model: load_latent(model, group)
            for model in MODEL_DIRS
        }

        for model_x, model_y in combinations(MODEL_DIRS, 2):
            X, names_x = loaded[model_x]
            Y, names_y = loaded[model_y]

            X, Y, names = match_identical_tiles(
                X,
                names_x,
                Y,
                names_y,
            )
            X, Y, names = subsample_matched(
                X,
                Y,
                names,
            )

            rows.append(
                {
                    "group": group,
                    "model_1": model_x,
                    "model_2": model_y,
                    "n_matched": len(names),
                    "linear_CKA": linear_cka(X, Y),
                }
            )

    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
