import numpy as np
import pandas as pd
import polars as pl


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


def generate_correlation_based_order_of_features_polars(
    feature_correlation: pl.DataFrame, correlation_treshold: int = 0.2
) -> pl.DataFrame:
    feature_correlation = feature_correlation.with_columns(
        [pl.col(col).abs().alias(col) for col in feature_correlation.columns]
    )
    feature_correlation = feature_correlation.with_columns(
        [
            pl.when(pl.col(col) < correlation_treshold)
            .then(0.0)
            .otherwise(pl.col(col))
            .alias(col)
            for col in feature_correlation.columns
        ]
    )

    arr = feature_correlation.to_numpy()
    np.fill_diagonal(arr, 0.0)
    feature_correlation = pl.DataFrame(arr, schema=feature_correlation.columns)
    return pl.DataFrame(
        {
            "feature": feature_correlation.columns,
            "count": feature_correlation.select((pl.all() != 0).sum()).row(0),
            "max": feature_correlation.max().row(0),
            "min": feature_correlation.select(
                pl.when(pl.all() != 0).then(pl.all()).otherwise(None).min()
            ).row(0),
            "sum": feature_correlation.sum().row(0),
        }
    ).sort(by=FEATURE_IMPORTANCE_ORDER)
