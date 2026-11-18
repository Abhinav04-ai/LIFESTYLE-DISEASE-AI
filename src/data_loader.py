from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, DISEASES

PIMA_COLUMNS = DISEASES["diabetes"]["features"] + ["Outcome"]
HEART_COLUMNS = DISEASES["heart"]["features"] + ["num"]


class DatasetError(ValueError):
    """Raised when a source dataset is missing or structurally invalid."""


def dataset_path(disease: str) -> Path:
    if disease not in DISEASES:
        raise ValueError(f"Unknown disease key: {disease}")
    return DATA_DIR / DISEASES[disease]["dataset"]


def load_dataset(disease: str, data_dir: Path | None = None) -> pd.DataFrame:
    if disease not in DISEASES:
        raise ValueError(f"Unknown disease key: {disease}")
    directory = Path(data_dir) if data_dir is not None else DATA_DIR
    path = directory / DISEASES[disease]["dataset"]
    if not path.is_file():
        raise DatasetError(
            f"Missing dataset file: {path}. Download the original dataset "
            f"and place it at this exact path. Source: "
            f"{DISEASES[disease]['source_url']}"
        )

    columns = PIMA_COLUMNS if disease == "diabetes" else HEART_COLUMNS
    try:
        frame = pd.read_csv(
            path, header=None, na_values="?", skipinitialspace=True
        )
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise DatasetError(f"Could not read {path}: {exc}") from exc

    if frame.empty or frame.shape[1] != len(columns):
        raise DatasetError(
            f"{path.name} must contain {len(columns)} comma-separated columns; "
            f"found {frame.shape[1]}."
        )
    frame.columns = columns

    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame[list(DISEASES[disease]["features"])].isna().all(axis=None):
        raise DatasetError(f"{path.name} has no usable numeric feature values.")
    if frame[DISEASES[disease]["target"]].isna().any():
        raise DatasetError(f"{path.name} contains missing or non-numeric target values.")
    if not np.isfinite(frame[DISEASES[disease]["target"]].to_numpy(dtype=float)).all():
        raise DatasetError(f"{path.name} contains non-finite target values.")

    if disease == "diabetes":
        targets = set(frame["Outcome"].unique())
        if not targets.issubset({0, 1}):
            raise DatasetError(
                f"Diabetes target values must be 0 or 1; found {sorted(targets)}."
            )
        y = frame["Outcome"].astype(int)
    else:
        targets = set(frame["num"].unique())
        if not targets.issubset({0, 1, 2, 3, 4}):
            raise DatasetError(
                f"Cleveland target values must be 0-4; found {sorted(targets)}."
            )
        y = (frame["num"] > 0).astype(int)

    counts = y.value_counts()
    if len(counts) != 2 or counts.min() < 5:
        raise DatasetError(
            "The dataset must contain at least five rows from each target class "
            "to support stratified five-fold evaluation."
        )
    return frame


def get_features_and_target(frame: pd.DataFrame, disease: str):
    """Return the notebook's feature columns and its binary target."""
    spec = DISEASES[disease]
    x = frame[spec["features"]].copy()
    if spec["zero_is_missing"]:
        x[spec["zero_is_missing"]] = x[spec["zero_is_missing"]].replace(0, np.nan)
    if disease == "heart":
        y = (frame[spec["target"]] > 0).astype(int)
    else:
        y = frame[spec["target"]].astype(int)
    return x, y
