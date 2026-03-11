from ctgan import CTGAN
import pandas as pd
from sklearn import preprocessing


class CTGANGenerator:
    TARGET = "target"

    def __init__(self, epochs: int = 300, preprocess: bool = False):
        self.epochs = epochs
        self.model = CTGAN(epochs=self.epochs)
        if preprocess:
            self._min_max_scaler = preprocessing.MinMaxScaler()
        else:
            self._min_max_scaler = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        data = X.copy()
        data[CTGANGenerator.TARGET] = y
        if self._min_max_scaler:
            scaled_data = self._min_max_scaler.fit_transform(data.values)
            data = pd.DataFrame(scaled_data, columns=data.columns)
        self.model.fit(data.astype(float), discrete_columns=[CTGANGenerator.TARGET])

    def generate(self, n_samples: int) -> pd.DataFrame:
        return self.model.sample(n_samples)

    def __call__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_samples: int,
        balance_classes: bool,
        **kwargs,
    ) -> tuple[pd.DataFrame, pd.Series]:
        self.fit(X_train, y_train)
        synthetic_data = self.generate(n_samples)
        if self._min_max_scaler:
            scaled_data = self._min_max_scaler.inverse_transform(synthetic_data.values)
            synthetic_data = pd.DataFrame(scaled_data, columns=synthetic_data.columns)
        return (
            synthetic_data.drop(columns=[CTGANGenerator.TARGET], axis=1),
            synthetic_data[CTGANGenerator.TARGET],
        )
