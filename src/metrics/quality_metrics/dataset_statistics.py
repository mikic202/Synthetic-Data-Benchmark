import pandas as pd
from scipy.stats import kstest


def calculate_dataset_statistics(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset_dsc = dataset.describe()
    dataset_dsc = pd.concat(
        [
            dataset_dsc,
            dataset.kurtosis().to_frame().T.set_index(pd.Index(["kurtosis"])),
            dataset.skew().to_frame().T.set_index(pd.Index(["skew"])),
            dataset.median().to_frame().T.set_index(pd.Index(["median"])),
        ]
    ).drop("count")

    train_kstest = kstest(dataset, "norm")
    dataset_dsc.loc["kstest_statistic"] = train_kstest.statistic
    dataset_dsc.loc["kstest_pvalue"] = train_kstest.pvalue

    return dataset_dsc
