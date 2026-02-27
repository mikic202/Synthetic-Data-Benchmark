import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


def measure_how_well_svn_distinguishes_real_data(
    real: pd.DataFrame, synthetic: pd.DataFrame, test_size: int = 0.2
) -> float:
    discriminator = XGBClassifier()
    combined_data = pd.concat([real, synthetic])
    datatype_labels = [1] * len(real) + [0] * len(synthetic)

    X_train, X_test, y_train, y_test = train_test_split(
        combined_data, datatype_labels, test_size=test_size, random_state=42
    )

    discriminator.fit(X_train, y_train)
    return discriminator.score(X_test, y_test)
