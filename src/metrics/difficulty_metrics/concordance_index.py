import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tabpfn import TabPFNRegressor
from tabicl import TabICLRegressor
from sklearn.linear_model import LinearRegression
import torch
from sklearn.preprocessing import StandardScaler
from multiprocessing import Process, Queue
import multiprocessing as mp
import torch
from sklearn.preprocessing import StandardScaler
from multiprocessing import Process, Queue
import multiprocessing as mp
from logging import getLogger
from sklearn.metrics import mean_squared_error
from sksurv.metrics import concordance_index_censored

logger = getLogger(__name__)

def measure_concordance_index(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int | float]],
    real_x: pd.DataFrame,
    real_y: list[int],
    model_class,
    **kwargs,
):
    index_values = []
    for synth_x, synth_y in zip(synthetic_x, synthetic_y):
        regressor = model_class(**kwargs).fit(synth_x, synth_y)
        predicted_y = regressor.predict(real_x)
        index_values.append((1 - float(concordance_index_censored(np.ones(len(real_y), dtype=bool), real_y, predicted_y)[0]),
                float(np.sqrt(mean_squared_error(real_y, predicted_y))),))
    return index_values


def measure_logistic_regresion_concordance_index(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    return measure_concordance_index(
        synthetic_x,
        synthetic_y,
        reral_x,
        real_y,
        LinearRegression,
        n_jobs=-1,
    )


def measure_random_forest_concordance_index(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    return measure_concordance_index(
        synthetic_x,
        synthetic_y,
        reral_x,
        real_y,
        RandomForestRegressor,
    )


def measure_tabpfn_concordance_index(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    return measure_concordance_index(
        synthetic_x,
        synthetic_y,
        reral_x,
        real_y,
        TabPFNRegressor,
        n_estimators=len(reral_x.columns) * 2,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )


def measure_tabicl_concordance_index(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    return measure_concordance_index(
        synthetic_x,
        synthetic_y,
        reral_x,
        real_y,
        TabICLRegressor,
        n_estimators=len(reral_x.columns) * 2,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )


def measure_xgb_concordance_index(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    n_estimators = len(reral_x.columns) * 2
    max_depth = max(len(reral_x.columns) // 5, 4)

    return measure_concordance_index(
        synthetic_x,
        synthetic_y,
        reral_x,
        real_y,
        XGBRegressor,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )


def random_forest_process(
    downstream_results_queue: Queue,
    synth_x: pd.DataFrame,
    synth_y: pd.DataFrame,
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
):
    downstream_results_queue.put(
        ("RF", measure_random_forest_concordance_index([synth_x], [synth_y], test_x, test_y)[0])
    )


def xgb_process(
    downstream_results_queue: Queue,
    synth_x: pd.DataFrame,
    synth_y: pd.DataFrame,
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
):
    downstream_results_queue.put(
        ("XGB", measure_xgb_concordance_index([synth_x], [synth_y], test_x, test_y)[0])
    )


def tabpfn_process(
    downstream_results_queue: Queue,
    synth_x: pd.DataFrame,
    synth_y: pd.DataFrame,
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
):
    downstream_results_queue.put(
        ("TabPFN", measure_tabpfn_concordance_index([synth_x], [synth_y], test_x, test_y)[0])
    )


def logistic_regression_process(
    downstream_results_queue: Queue,
    synth_x: pd.DataFrame,
    synth_y: pd.DataFrame,
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
):
    downstream_results_queue.put(
        ("LR", measure_logistic_regresion_concordance_index([synth_x], [synth_y], test_x, test_y)[0])
    )


def tabicl_process(
    downstream_results_queue: Queue,
    synth_x: pd.DataFrame,
    synth_y: pd.DataFrame,
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
):
    downstream_results_queue.put(
        ("TabICL", measure_tabicl_concordance_index([synth_x], [synth_y], test_x, test_y)[0])
    )


def calculate_concordance_index(
    synth_x: pd.DataFrame,
    synth_y: pd.DataFrame,
    real_x: pd.DataFrame,
    real_y: pd.DataFrame,
) -> list[float]:
    mp.set_start_method("spawn", force=True)
    downstream_results_queue = Queue()
    downstream_jobs = []

    downstream_jobs.append(
        Process(
            target=random_forest_process,
            args=(downstream_results_queue, synth_x, synth_y, real_x, real_y),
            name="RandomForestClassifier",
        )
    )
    downstream_jobs.append(
        Process(
            target=logistic_regression_process,
            args=(downstream_results_queue, synth_x, synth_y, real_x, real_y),
            name="LogisticRegression",
        )
    )
    downstream_jobs.append(
        Process(
            target=xgb_process,
            args=(downstream_results_queue, synth_x, synth_y, real_x, real_y),
            name="XGBClassifier",
        )
    )
    downstream_jobs.append(
        Process(
            target=tabpfn_process,
            args=(downstream_results_queue, synth_x, synth_y, real_x, real_y),
            name="TabPFNClassifier",
        )
    )
    downstream_jobs.append(
        Process(
            target=tabicl_process,
            args=(downstream_results_queue, synth_x, synth_y, real_x, real_y),
            name="TabICLClassifier",
        )
    )

    for job in downstream_jobs:
        job.start()
    for job in downstream_jobs:
        logger.debug(f"Waiting for {job.name} to finish...")
        job.join()
    return {
        model_name: accuracy
        for model_name, accuracy in sorted(
            downstream_results_queue.get() for _ in range(len(downstream_jobs))
        )
    }
