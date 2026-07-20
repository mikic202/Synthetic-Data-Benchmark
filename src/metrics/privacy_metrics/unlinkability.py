import pandas as pd
from scipy.stats import entropy
import numpy as np


def calculate_unlinkability(
    synthetic_dataset: pd.DataFrame,
    real_dataset: pd.DataFrame,
    identifier_atributes: list[str] | None = None,
):
    if not identifier_atributes:
        identifier_atributes = synthetic_dataset.columns.tolist()

    distances = np.array(
        [
            np.linalg.norm(abs(real - synthetic))
            for real in real_dataset[identifier_atributes].values
            for synthetic in synthetic_dataset[identifier_atributes].values
        ]
    )
    probabilities = np.exp(-distances)
    if np.sum(probabilities) <= 1e-30:
        return 0.0
    return entropy(probabilities / np.sum(probabilities), base=2)
