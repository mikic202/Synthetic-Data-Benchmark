import pandas as pd
import numpy as np
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
