import numpy as np
import pandas as pd


FEATURE_IMPORTANCE_ORDER = ["count", "max", "min", "sum"]


def generate_correlation_based_order_of_features_in_dataset(
    dataset: pd.DataFrame, correlation_treshold: int = 0.2
) -> list[str]:
    return generate_correlation_based_order_of_features(
        dataset.corr(), correlation_treshold
    ).index.tolist()


def generate_correlation_based_order_of_features(
    feature_correlation: pd.DataFrame, correlation_treshold: int = 0.2
) -> pd.DataFrame:

    feature_correlation = abs(feature_correlation)
    mask = feature_correlation < correlation_treshold
    feature_correlation[mask] = 0.0
    np.fill_diagonal(feature_correlation.values, 0.0)
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
