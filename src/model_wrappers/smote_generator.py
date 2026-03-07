import pandas as pd
from imblearn.over_sampling import SMOTE
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance


class SmoteGenerator:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_samples: int,
        balance_classes: bool,
        **kwargs,
    ):
        classes, counts = np.unique(y_train, return_counts=True)
        n_classes = len(classes)
        per_class = n_samples // n_classes

        generator = SMOTE(
            sampling_strategy={
                cls: counts[classes == cls][0] + per_class for cls in classes
            },
            **kwargs,
        )
        x_synth, y_synth = generator.fit_resample(X_train, y_train)

        return x_synth[-n_samples:], y_synth[-n_samples:]


class SmoterGenerator:
    def __init__(self, k_nearest: int = 5) -> None:
        self.k_nearest = k_nearest

    def __call__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_samples: int,
        balance_classes: bool,
        **kwargs,
    ):
        new_examples = []
        train = X_train.copy()
        train["target"] = y_train
        knn = NearestNeighbors(n_neighbors=self.k_nearest + 1)
        knn.fit(train)
        for _ in range(n_samples):
            random_sample = train.sample(n=1)
            neighbours = knn.kneighbors(random_sample, return_distance=False)[0][1:]
            random_neighbour = random.sample(list(neighbours), 1)[0]
            random_neighbour = train.iloc[[random_neighbour]]
            new_example = {}
            for atribute in X_train.columns:
                if X_train[atribute].dtype is float:
                    new_example[atribute] = random_sample[atribute].values[
                        0
                    ] + random.random() * (
                        random_sample[atribute].values[0] - train[atribute].values[0]
                    )
                else:
                    new_example[atribute] = random.sample(
                        [
                            random_sample[atribute].values[0],
                            train[atribute].values[0],
                        ],
                        k=1,
                    )[0]
            distance_to_real = distance.euclidean(
                list(new_example.values()),
                random_sample.drop("target", axis=1).to_numpy()[0],
            )
            distance_to_neighbour = distance.euclidean(
                list(new_example.values()),
                random_neighbour.drop("target", axis=1).to_numpy()[0],
            )
            new_example["target"] = (
                distance_to_real * random_sample["target"].values[0]
                + distance_to_neighbour * random_sample["target"].values[0]
            ) / (distance_to_neighbour + distance_to_real)
            new_examples.append(new_example)
        new_examples = pd.DataFrame(new_examples)
        return new_examples.drop("target", axis=1), new_examples["target"]
