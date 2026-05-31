import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit
import numpy as np


def calculate_xgb_descrimination(
    real: pd.DataFrame, synthetic: pd.DataFrame, test_size: int = 0.25
) -> float:
    discriminator = XGBClassifier(max_depth=3, n_estimators=4, max_bin=2)
    train_indecies, test_indecies = next(
        ShuffleSplit(n_splits=1, test_size=test_size).split(
            np.arange(min(len(real), len(synthetic)))
        )
    )
    X_train = pd.concat([real.iloc[train_indecies], synthetic.iloc[train_indecies]])
    y_train = [1] * len(train_indecies) + [0] * len(train_indecies)
    X_test = pd.concat([real.iloc[test_indecies], synthetic.iloc[test_indecies]])
    y_test = [1] * len(test_indecies) + [0] * len(test_indecies)

    discriminator.fit(X_train, y_train)
    return discriminator.score(X_test, y_test)
