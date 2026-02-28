from tabpfn_extensions import unsupervised
from tabpfn import TabPFNClassifier, TabPFNRegressor
import pandas as pd
import torch
from typing import Callable


class FullTabpfnGen(unsupervised.TabPFNUnsupervisedModel):
    def __init__(
        self,
        device,
        column_order_getter: Callable[[pd.DataFrame, bool], list[str]] | None = None,
    ):
        super().__init__(
            tabpfn_clf=TabPFNClassifier(device=device),
            tabpfn_reg=TabPFNRegressor(device=device),
        )
        self._column_order_getter = column_order_getter

    def __call__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        n_samples: int,
        attribute_names: list[str] | None = None,
        indices: list[int] | None = None,
        temp=1.0,
        **kwargs
    ):

        train_data = X_train.copy()
        train_data["target"] = y_train
        data = torch.tensor(train_data.to_numpy())

        if self._column_order_getter:
            feature_order = self._column_order_getter(train_data)
        else:
            feature_order = train_data.columns.to_list()
        if indices is None or attribute_names is None:
            categorical_features = feature_order

        else:
            feature_names = [attribute_names[i] for i in indices]
            categorical_features = [
                feature_names.index(name)
                for name in attribute_names
                if name in feature_names
            ]
        self.set_categorical_features(categorical_features)
        self.fit(data)

        synthetic_data = self.generate_synthetic_data(
            n_samples=n_samples,
            t=temp,
        )
        synthetic_data = pd.DataFrame(synthetic_data, columns=train_data.columns)
        return (
            synthetic_data.drop("target", axis=1),
            (
                (synthetic_data["target"].round() - 1).to_list()
                if min(synthetic_data["target"].round()) > 0
                else synthetic_data["target"].round().to_list()
            ),
        )
