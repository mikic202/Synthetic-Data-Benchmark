from ctgan import CTGAN
import pandas as pd


class CTGANGenerator:
    TARGET = "target"

    def __init__(self, epochs: int = 300):
        self.epochs = epochs
        self.model = CTGAN(epochs=self.epochs)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        data = X.copy()
        data[CTGANGenerator.TARGET] = y
        self.model.fit(data.astype(float), discrete_columns=[CTGANGenerator.TARGET])

    def generate(self, n_samples: int) -> tuple[pd.DataFrame, pd.Series]:
        samples = self.model.sample(n_samples)
        return (
            samples.drop(columns=[CTGANGenerator.TARGET]),
            samples[CTGANGenerator.TARGET],
        )

    def __call__(
        self, X_train: pd.DataFrame, y_train: pd.Series, n_samples: int, **kwargs
    ) -> tuple[pd.DataFrame, pd.Series]:
        self.fit(X_train, y_train)
        return self.generate(n_samples)
