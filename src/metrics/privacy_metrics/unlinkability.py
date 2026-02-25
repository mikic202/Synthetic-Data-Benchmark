import pandas as pd
from scipy.stats import entropy
import math


def calculate_unlinkability(
    synthetic_dataset: pd.DataFrame,
    real_dataset: pd.DataFrame,
    identifier_atributes: list[str] | None = None,
):
    if not identifier_atributes:
        identifier_atributes = synthetic_dataset.columns.tolist()

    set_of_pairs = [
        [(real, synthetic) for real in real_dataset[identifier_atributes].values]
        for synthetic in synthetic_dataset[identifier_atributes].values
    ][0]

    total_unlinkability = 0
    for pair in set_of_pairs:
        pair_unlinkability = sum(entropy(pair)) / len(pair)
        total_unlinkability += math.log2(pair_unlinkability) * pair_unlinkability
    return -total_unlinkability
