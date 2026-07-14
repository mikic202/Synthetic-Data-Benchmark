import pandas as pd
from src.metrics import (
    NUMBER_OF_UNIQUE_ELEMENTS_FOR_CLASIFICATION,
)
import numpy as np
import scipy.stats as stats


class RandomGenerator:
    def __init__(self, only_sample_real: bool = True) -> None:
        self.__only_sample_real = only_sample_real

    def __call__(self,
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        n_samples: int,
        balance_classes: bool,
        **kwargs,):
        synth_x = pd.DataFrame()
        for column in x_train.columns:
            if len(x_train[column].unique()) <= NUMBER_OF_UNIQUE_ELEMENTS_FOR_CLASIFICATION:
                synth_x[column] = self._generate_discrete_column(x_train[column], n_samples)
            else:
                synth_x[column] = self._generate_continous_column(x_train[column], n_samples)
        if len(np.unique(y_train)) <= NUMBER_OF_UNIQUE_ELEMENTS_FOR_CLASIFICATION:
            synth_y = self._generate_discrete_column(y_train, n_samples)
        else:
            synth_y = self._generate_continous_column(y_train, n_samples)
        return synth_x, synth_y

    def _generate_discrete_column(self, column_data: list, n_samples: int) -> list:
        elements, counts = np.unique(column_data, return_counts=True)
        probabilities = counts / counts.sum()
        rng = np.random.default_rng()
        return rng.choice(elements, size=n_samples, p=probabilities)

    def _generate_continous_column(self, column_data: list, n_samples: int) -> list:
        if not self.__only_sample_real:
            return stats.gaussian_kde(column_data).resample(size=n_samples).flatten()
        elements, counts = np.unique(column_data, return_counts=True)
        probabilities = counts / counts.sum()
        rng = np.random.default_rng()
        return rng.choice(elements, size=n_samples, p=probabilities)