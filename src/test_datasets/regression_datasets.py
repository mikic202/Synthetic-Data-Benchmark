import pandas as pd
import openml
from sklearn.model_selection import train_test_split
import kagglehub


REGRESION_TARGET = "target"


def get_heart_failure_clinical_regresion_dataset(
    test_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset, _, _, _ = openml.datasets.get_dataset(46612).get_data(
        dataset_format="dataframe"
    )
    dataset = dataset.rename(columns={"platelets": REGRESION_TARGET}).astype("float32")
    return train_test_split(dataset, test_size=test_size, random_state=42)


def get_superconduct_regression_dataset(
    test_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset, _, _, _ = openml.datasets.get_dataset(44148).get_data(
        dataset_format="dataframe"
    )
    dataset = dataset.rename(columns={"criticaltemp": REGRESION_TARGET}).astype(
        "float32"
    )
    return train_test_split(dataset, test_size=test_size, random_state=42)


def get_sleep_deprivation_and_cognitive_performance_regression_dataset(
    test_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset, _, _, _ = openml.datasets.get_dataset(46754).get_data(
        dataset_format="dataframe"
    )
    dataset = dataset.drop(columns=["Participant_ID"], axis=1)
    dataset["Gender"] = dataset["Gender"].factorize()[0]
    dataset = dataset.rename(columns={"Stress_Level": REGRESION_TARGET})
    return train_test_split(dataset, test_size=test_size, random_state=42)


def get_house_prices_regression_dataset(
    test_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset, _, _, _ = openml.datasets.get_dataset(42563).get_data(
        dataset_format="dataframe"
    )
    nominal_features = [
        "MSZoning",
        "Street",
        "Alley",
        "LotShape",
        "LandContour",
        "Utilities",
        "LotConfig",
        "LandSlope",
        "Neighborhood",
        "Condition1",
        "Condition2",
        "BldgType",
        "HouseStyle",
        "RoofStyle",
        "RoofMatl",
        "Exterior1st",
        "Exterior2nd",
        "MasVnrType",
        "ExterQual",
        "ExterCond",
        "Foundation",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "Heating",
        "HeatingQC",
        "CentralAir",
        "Electrical",
        "KitchenQual",
        "Functional",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "PavedDrive",
        "PoolQC",
        "Fence",
        "MiscFeature",
        "SaleType",
        "SaleCondition",
    ]
    for feature_to_factorize in nominal_features:
        dataset[feature_to_factorize] = dataset[feature_to_factorize].factorize()[0]
    dataset = dataset.rename(columns={"SalePrice": REGRESION_TARGET})
    dataset = dataset.dropna()
    return train_test_split(dataset, test_size=test_size, random_state=42)


def get_cardiovascular_dataset(test_size: float = 0.1):
    path = kagglehub.dataset_download("sulianova/cardiovascular-disease-dataset")
    dataset = pd.read_csv(path + "/cardio_train.csv", sep=";")
    dataset = dataset.rename(columns={"cardio": REGRESION_TARGET}).astype("float32")
    return train_test_split(dataset, test_size=test_size, random_state=42)


def get_abalone_dataset(test_size: float = 0.1):
    path = kagglehub.dataset_download("rodolfomendes/abalone-dataset")
    print(path)
    dataset = pd.read_csv(path + "/abalone.csv", sep=";")
    dataset = dataset.rename(columns={"Rings": REGRESION_TARGET}).astype("float32")
    return train_test_split(dataset, test_size=test_size, random_state=42)
