import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np


def calculate_distance_to_nearest_record(
    synthetic_dataset: pd.DataFrame,
    identifier_atributes: list[str] | None = None,
) -> dict[str, float]:
    if not identifier_atributes:
        identifier_atributes = synthetic_dataset.columns.tolist()

    model = NearestNeighbors(n_neighbors=2)
    model.fit(synthetic_dataset[identifier_atributes])
    distances = model.kneighbors(synthetic_dataset[identifier_atributes])[0][:, 1]
    return {
        "mean": np.mean(distances),
        "std": np.std(distances),
        "median": np.median(distances),
    }


def calculate_distance_to_nearest_real_record(
    synthetic_dataset: pd.DataFrame,
    real_dataset: pd.DataFrame,
    identifier_atributes: list[str] | None = None,
) -> dict[str, float]:
    if not identifier_atributes:
        identifier_atributes = synthetic_dataset.columns.tolist()

    model = NearestNeighbors(n_neighbors=1)
    model.fit(real_dataset[identifier_atributes])
    distances, _ = model.kneighbors(synthetic_dataset[identifier_atributes])
    return {
        "mean": np.mean(distances),
        "std": np.std(distances),
        "median": np.median(distances),
    }
