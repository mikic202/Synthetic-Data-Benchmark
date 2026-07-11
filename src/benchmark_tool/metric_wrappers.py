from abc import ABC, abstractmethod
import pandas as pd

from src import metrics


class MetricWrapper(ABC):
    @staticmethod
    @abstractmethod
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
        return int(metrics.calculate_k_anonimity_for_datset(synthetic))


class KAnonimityWithReal(MetricWrapper):
    @staticmethod
    def __call__(synthetic: pd.DataFrame, real_train: pd.DataFrame, *args, **kwargs):
        return int(
            metrics.calculate_relative_k_anonimity_for_dataset(real_train, synthetic)
        )


class Unlinkability(MetricWrapper):
    @staticmethod
    def __call__(synthetic: pd.DataFrame, real_train: pd.DataFrame, *args, **kwargs):
        return float(metrics.calculate_unlinkability(synthetic, real_train))


class DistanceToNearestRealNeighbour(MetricWrapper):
    @staticmethod
    def __call__(synthetic: pd.DataFrame, real_train: pd.DataFrame, *args, **kwargs):
        return metrics.calculate_distance_to_nearest_real_record(synthetic, real_train)


class DistanceToNearestNeighbour(MetricWrapper):
    @staticmethod
    def __call__(synthetic: pd.DataFrame, *args, **kwargs):
        return metrics.calculate_distance_to_nearest_record(synthetic)


class DatasetStatistics(MetricWrapper):
    @staticmethod
    def __call__(synthetic: pd.DataFrame, *args, **kwargs):
        return metrics.calculate_dataset_statistics(synthetic)


class MinimalTree(MetricWrapper):
    @staticmethod
    def __call__(
        synthetic: pd.DataFrame, real_test: pd.DataFrame, target: str, *args, **kwargs
    ):
        synth_x = synthetic.drop(target, axis=1)
        synth_y = synthetic[target]
        test_x = real_test.drop(target, axis=1)
        test_y = real_test[target]
        return metrics.calculate_ralation_between_dree_depth_and_accuaracy(
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
        if (
            test_y.nunique() <= metrics.NUMBER_OF_UNIQUE_ELEMENTS_FOR_CLASIFICATION
            and synth_y.nunique() <= metrics.NUMBER_OF_UNIQUE_ELEMENTS_FOR_CLASIFICATION
        ):
            return metrics.calculate_auc(synth_x, synth_y, test_x, test_y)
        else:
            return metrics.calculate_concordance_index(synth_x, synth_y, test_x, test_y)


class ConvexHull(MetricWrapper):
    @staticmethod
    def __call__(
        synthetic: pd.DataFrame,
        real_train: pd.DataFrame,
        real_test: pd.DataFrame,
        *args,
        **kwargs,
    ):
        return float(
            metrics.calculate_convex_hull_coverage(
                pd.concat([real_test, real_train]), synthetic
            )
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
        return metrics.calculate_xgb_descrimination(
            pd.concat([real_test, real_train]), synthetic
        )


class Identity(MetricWrapper):
    @staticmethod
    def __call__(
        synthetic: pd.DataFrame,
        real_train: pd.DataFrame,
        *args,
        **kwargs,
    ):
        return metrics.get_number_of_real_examples_in_synthetic(
           real_train, synthetic
        )