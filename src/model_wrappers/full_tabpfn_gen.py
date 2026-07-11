from tabpfn_extensions import unsupervised
from tabpfn import TabPFNClassifier, TabPFNRegressor
import pandas as pd
import torch
import os
import numpy as np
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
        train_data.insert(0, "target", y_train)
        # train_data["target"] = y_train
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

        assert hasattr(
            self,
            "X_",
        ), "You need to fit the model before generating synthetic data"

        fast_mode = os.environ.get("FAST_TEST_MODE", "0") == "1"

        actual_n_permutations = 1 if fast_mode else 3

        X = torch.zeros(n_samples, data.shape[1]) * np.nan
        X[:, 0] = torch.tensor(y_train)

        synthetic_data = self.impute_(
            X,
            t=temp,
            condition_on_all_features=False,
            n_permutations=actual_n_permutations,
            fast_mode=fast_mode,
        )


        synthetic_data = pd.DataFrame(synthetic_data, columns=train_data.columns)
        rounded_target = synthetic_data["target"].round()
        return (
            synthetic_data.drop("target", axis=1),
            (
                (rounded_target - 1).to_list()
                if min(rounded_target) > 0
                else (rounded_target + 1).to_list() if min(rounded_target) < 0
                else rounded_target.to_list()
            ),
        )
