import pandas as pd
from imblearn.over_sampling import SMOTE
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
from random import choices


class RealDataGenerator:
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
        random_indecies = choices(range(len(X_train)), k=n_samples)
        return (
            X_train.iloc[random_indecies],
            np.array(y_train)[random_indecies].tolist(),
        )
