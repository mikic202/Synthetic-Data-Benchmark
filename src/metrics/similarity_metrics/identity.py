import pandas as pd
import numpy as np


def get_identical_samples_ids(samples: pd.DataFrame, reference: pd.DataFrame):
    combined_dfs = pd.concat([reference, samples])
    return (np.flatnonzero(combined_dfs.duplicated()) - len(reference)).to_list()


def get_number_of_real_examples_in_synthetic(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> int:
    return len(get_identical_samples_ids(synthetic_data, real_data))