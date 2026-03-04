from abc import ABC, abstractmethod
import pandas as pd
from src.metrics.privacy_metrics import (
    k_anonimity,
    unlinkability,
    distance_to_nearest_neighbour,
)
from src.metrics.quality_metrics import dataset_statistics
from src.metrics.difficulty_metrics import minimal_tree, model_auc
from src.metrics.similarity_metrics import convex_hull, discriminator


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


class MinimalTree(MetricWrapper):
    @staticmethod
    def __call__(
        synthetic: pd.DataFrame, real_test: pd.DataFrame, target: str, *args, **kwargs
    ):
        synth_x = synthetic.drop(target, axis=1)
        synth_y = synthetic[target]
        test_x = real_test.drop(target, axis=1)
        test_y = real_test[target]
        return minimal_tree.calculate_ralation_between_dree_depth_and_accuaracy(
            synth_x, synth_y, test_x, test_y
        )


class ModelAuc(MetricWrapper):
    @staticmethod
    def __call__(
        synthetic: pd.DataFrame, real_test: pd.DataFrame, target: str, *args, **kwargs
    ):
        synth_x = synthetic.drop(target, axis=1)
        synth_y = synthetic[target]
        test_x = real_test.drop(target, axis=1)
        test_y = real_test[target]
        return model_auc.calculate_auc(synth_x, synth_y, test_x, test_y)


class ConvexHull(MetricWrapper):
    @staticmethod
    def __call__(
        synthetic: pd.DataFrame,
        real_train: pd.DataFrame,
        real_test: pd.DataFrame,
        *args,
        **kwargs,
    ):
        return convex_hull.calculate_convex_hull_coverage(
            pd.concat([real_test, real_train]), synthetic
        )


class Discrimination(MetricWrapper):
    @staticmethod
    def __call__(
        synthetic: pd.DataFrame,
        real_train: pd.DataFrame,
        real_test: pd.DataFrame,
        *args,
        **kwargs,
    ):
        return discriminator.measure_how_well_svn_distinguishes_real_data(
            pd.concat([real_test, real_train]), synthetic
        )
