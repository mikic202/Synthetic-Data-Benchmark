from abc import ABC, abstractmethod
import pandas as pd
from src.metrics.privacy_metrics import (
    k_anonimity,
    unlinkability,
    distance_to_nearest_neighbour,
)
from src.metrics.quality_metrics import dataset_statistics


class MetricWrapper(ABC):
    @abstractmethod
    @staticmethod
    def __call__(
        synthetic: pd.DataFrame,
        real_train: pd.DataFrame,
        real_test: pd.DataFrame,
        target: str,
    ):
        pass


class KAnonimity(MetricWrapper):
    @staticmethod
    def __call__(synthetic: pd.DataFrame, *args, **kwargs):
        return k_anonimity.calculate_k_anonimity_for_datset(synthetic)


class Unlinkability(MetricWrapper):
    @staticmethod
    def __call__(synthetic: pd.DataFrame, real_train: pd.DataFrame, *args, **kwargs):
        return unlinkability.calculate_unlinkability(synthetic, real_train)


class DistanceToNearestNeighbour(MetricWrapper):
    @staticmethod
    def __call__(synthetic: pd.DataFrame, real_train: pd.DataFrame, *args, **kwargs):
        return distance_to_nearest_neighbour.calculate_distance_toNearest_record(
            synthetic, real_train
        )


class DatasetStatistics(MetricWrapper):
    @staticmethod
    def __call__(synthetic: pd.DataFrame, *args, **kwargs):
        return dataset_statistics.calculate_dataset_statistics(synthetic)
