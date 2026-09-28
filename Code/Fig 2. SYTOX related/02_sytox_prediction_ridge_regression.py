"""Ridge regression for predicting SYTOX burden from latent representations."""

import os
import re
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("data")
LATENT_DIR = Path("latent_vectors")
OUTPUT_DIR = Path("outputs") / "ridge_regression"

SYTOX_CSV = DATA_DIR / "sytox_cell_death.csv"
TARGET_COLUMN = "intensity_per_pixel_mean_3std"

DISEASE_GROUPS = ["GBA", "PINK1", "SNCA"]

MODEL_PATHS = {
    "ResNet18": Path("path/to/resnet18_latent_vectors"),
    "SupMoCo": Path("path/to/supmoco_latent_vectors"),
}

TEST_SIZE = 0.20
SPLIT_RANDOM_SEED = 42
INNER_CV_SPLITS = 5
ALPHAS = np.logspace(-4, 5, 19)
USE_L2_NORMALIZATION = True

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_tile_id(value):
    value = os.path.basename(str(value))
    value = re.sub(
        r"\.(tif|tiff|png|jpg|jpeg|npy)$",
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


def extract_source_image_id(tile_id):
    tile_id = normalize_tile_id(tile_id)
    return re.sub(
        r"_x\d+_y\d+.*$",
        "",
        tile_id,
        flags=re.IGNORECASE,
    )


def extract_well_group(tile_id):
    tile_id = normalize_tile_id(tile_id)
    match = re.search(
        r"(.+?_r\d+c\d+)",
        tile_id,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1)
    return extract_source_image_id(tile_id)


def l2_normalize_rows(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def find_existing_file(directory, candidate_names):
    directory = Path(directory)
    for filename in candidate_names:
        path = directory / filename
        if path.exists():
            return path
    return None


def find_latent_files(latent_dir, disease):
    latent_candidates = [
        f"{disease}_latent_raw.npy",
        f"{disease}_latents.npy",
        f"{disease}_latent.npy",
        f"{disease}_latent_vectors.npy",
        f"{disease}_features.npy",
    ]
    filename_candidates = [
        f"{disease}_filenames.npy",
        f"{disease}_filename.npy",
        f"{disease}_names.npy",
    ]

    latent_file = find_existing_file(latent_dir, latent_candidates)
    filename_file = find_existing_file(latent_dir, filename_candidates)

    if latent_file is None:
        matches = [
            p for p in sorted(Path(latent_dir).glob(f"{disease}*latent*.npy"))
            if "filename" not in p.name.lower()
        ]
        if matches:
            latent_file = matches[0]

    if filename_file is None:
        matches = sorted(Path(latent_dir).glob(f"{disease}*filename*.npy"))
        if matches:
            filename_file = matches[0]

    return latent_file, filename_file


def load_sytox_data():
    df = pd.read_csv(SYTOX_CSV)

    required = {"filename", "folder", TARGET_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["tile_id"] = df["filename"].apply(normalize_tile_id)
    df["target"] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df[df["folder"].isin(DISEASE_GROUPS)].copy()
    df = df[np.isfinite(df["target"])].copy()

    if df["tile_id"].duplicated(keep=False).any():
        df = (
            df.groupby(["tile_id", "folder"], as_index=False)
            .agg(target=("target", "mean"))
        )

    return df


def load_matched_data(latent_dir, disease, tile_to_target):
    latent_file, filename_file = find_latent_files(latent_dir, disease)

    if latent_file is None or filename_file is None:
        raise FileNotFoundError(f"Missing latent or filename file: {latent_dir}")

    latent_vectors = np.load(latent_file, mmap_mode="r")
    filenames = np.load(filename_file, allow_pickle=True)

    if len(latent_vectors) != len(filenames):
        raise ValueError("Latent vectors and filenames have different lengths.")

    tile_ids = np.array([normalize_tile_id(x) for x in filenames])
    targets = np.array(
        [tile_to_target.get(tile_id, np.nan) for tile_id in tile_ids],
        dtype=float,
    )

    matched = np.isfinite(targets)

    X = np.asarray(latent_vectors[matched], dtype=np.float32)
    y = targets[matched].astype(np.float64)
    tile_ids = tile_ids[matched]
    groups = np.array([extract_well_group(x) for x in tile_ids])

    if USE_L2_NORMALIZATION:
        X = l2_normalize_rows(X)

    return X, y, groups, tile_ids


def make_group_split(X, y, groups, tile_ids):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=SPLIT_RANDOM_SEED,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    groups_train = groups[train_idx]
    groups_test = groups[test_idx]

    overlap = set(groups_train) & set(groups_test)
    if overlap:
        raise RuntimeError("Train/test group overlap detected.")

    return {
        "X_train": X[train_idx],
        "X_test": X[test_idx],
        "y_train": y[train_idx],
        "y_test": y[test_idx],
        "groups_train": groups_train,
        "groups_test": groups_test,
        "tile_ids_train": tile_ids[train_idx],
        "tile_ids_test": tile_ids[test_idx],
    }


def fit_and_evaluate_ridge(split_data):
    X_train = split_data["X_train"]
    X_test = split_data["X_test"]
    y_train = split_data["y_train"]
    y_test = split_data["y_test"]
    groups_train = split_data["groups_train"]

    n_cv_splits = min(
        INNER_CV_SPLITS,
        len(np.unique(groups_train)),
    )
    if n_cv_splits < 2:
        raise ValueError("Not enough training groups for GroupKFold.")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(max_iter=10000)),
    ])

    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"ridge__alpha": ALPHAS},
        scoring="neg_mean_squared_error",
        cv=GroupKFold(n_splits=n_cv_splits),
        n_jobs=-1,
        refit=True,
    )

    search.fit(
        X_train,
        y_train,
        groups=groups_train,
    )

    y_test_pred = search.best_estimator_.predict(X_test)
    test_rho, test_p = spearmanr(y_test, y_test_pred)

    return {
        "best_alpha": float(search.best_params_["ridge__alpha"]),
        "test_r2": float(r2_score(y_test, y_test_pred)),
        "test_spearman_rho": float(test_rho),
        "test_spearman_p": float(test_p),
        "test_rmse": float(
            np.sqrt(mean_squared_error(y_test, y_test_pred))
        ),
        "y_test_pred": y_test_pred,
    }


def main():
    sytox = load_sytox_data()
    tile_to_target = dict(zip(sytox["tile_id"], sytox["target"]))

    results = []
    predictions = []

    for model_name, latent_dir in MODEL_PATHS.items():
        for disease in DISEASE_GROUPS:
            X, y, groups, tile_ids = load_matched_data(
                latent_dir,
                disease,
                tile_to_target,
            )

            split_data = make_group_split(
                X,
                y,
                groups,
                tile_ids,
            )
            result = fit_and_evaluate_ridge(split_data)

            results.append({
                "model": model_name,
                "disease": disease,
                "best_alpha": result["best_alpha"],
                "test_r2": result["test_r2"],
                "test_spearman_rho": result["test_spearman_rho"],
                "test_spearman_p": result["test_spearman_p"],
                "test_rmse": result["test_rmse"],
                "n_train": len(split_data["y_train"]),
                "n_test": len(split_data["y_test"]),
                "n_groups_train": len(np.unique(split_data["groups_train"])),
                "n_groups_test": len(np.unique(split_data["groups_test"])),
            })

            for tile_id, group_id, observed, predicted in zip(
                split_data["tile_ids_test"],
                split_data["groups_test"],
                split_data["y_test"],
                result["y_test_pred"],
            ):
                predictions.append({
                    "model": model_name,
                    "disease": disease,
                    "tile_id": tile_id,
                    "well_group": group_id,
                    "observed": float(observed),
                    "predicted": float(predicted),
                })

    pd.DataFrame(results).to_csv(
        OUTPUT_DIR / "ridge_test_results.csv",
        index=False,
    )
    pd.DataFrame(predictions).to_csv(
        OUTPUT_DIR / "ridge_test_predictions.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
