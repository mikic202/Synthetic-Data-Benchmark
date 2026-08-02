import numpy as np
import pandas as pd
from src.feature_order.feature_order_cache import get_feature_order_from_cache, save_feature_order_to_cache
from pathlib import Path
import os


FEATURE_IMPORTANCE_ORDER = ["count", "max", "min", "sum"]


def generate_correlation_based_order_of_features_in_dataset(
    dataset: pd.DataFrame, correlation_treshold: int = 0.2
) -> list[str]:
    cache_file = Path(os.path.abspath(__file__)).parent/"tmp/correlation.json"
    if cached_feature_order := get_feature_order_from_cache(list(dataset.columns), cache_file):
        return cached_feature_order
    feature_order = generate_correlation_based_order_of_features(
        dataset.corr().copy(), correlation_treshold
    ).index.tolist()
    save_feature_order_to_cache(feature_order, cache_file)
    return feature_order


def generate_correlation_based_order_of_features(
    feature_correlation: pd.DataFrame, correlation_treshold: int = 0.2
) -> pd.DataFrame:

    feature_correlation = abs(feature_correlation)
    mask = feature_correlation < correlation_treshold
    feature_correlation[mask] = 0.0
    feature_correlation = feature_correlation.mask(np.eye(len(feature_correlation), dtype=bool), 0.0)
    return (
        pd.DataFrame(
            {
                "count": (feature_correlation > 0).sum(),
                "max": feature_correlation.max(),
                "min": feature_correlation.mask(feature_correlation <= 0).min(),
                "sum": feature_correlation.sum(),
            }
        )
        .fillna(0)
        .sort_values(by=FEATURE_IMPORTANCE_ORDER)
    )
