import pandas as pd
from imblearn.over_sampling import SMOTE
import numpy as np


class SmoteGenerator:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_samples: int,
        balance_classes: bool,
        **kwargs
    ):
        classes, counts = np.unique(y_train, return_counts=True)
        n_classes = len(classes)
        per_class = n_samples // n_classes

        generator = SMOTE(
            sampling_strategy={
                cls: counts[classes == cls][0] + per_class for cls in classes
            },
            **kwargs
        )
        x_synth, y_synth = generator.fit_resample(X_train, y_train)

        return x_synth[-n_samples:], y_synth[-n_samples:]
