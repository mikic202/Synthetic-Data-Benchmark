import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tabpfn import TabPFNRegressor
from sklearn.linear_model import LinearRegression
import torch
from sklearn.preprocessing import StandardScaler
from multiprocessing import Process, Queue
import multiprocessing as mp
from logging import getLogger
from src.metrics.difficulty_metrics.minimal_tree import (
    NUMBER_OF_UNIQUE_ELEMENTS_FOR_CLASIFICATION,
)


logger = getLogger(__name__)


NUMBER_OF_UNIQUE_ELEMENTS_FOR_BINARY_CLASIFICATION = 2


def measure_rroc_aoc(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int | float]],
    reral_x: pd.DataFrame,
    real_y: list[int],
    model_class,
    **kwargs,
):
    areas_under_curve = []
    for synt_x, synth_y in zip(synthetic_x, synthetic_y):
        scaler = StandardScaler()
        scaler.fit(synt_x)
        scalled_synth_x = scaler.transform(synt_x)
        scalled_real_x = scaler.transform(reral_x)
        regressor = model_class(**kwargs).fit(scalled_synth_x, synth_y)
        predicted_y = regressor.predict(scalled_real_x)
        errors = predicted_y - real_y
        areas_under_curve.append(np.var(errors) / 2.0 * (len(errors) ** 2))
    return areas_under_curve


def measure_linear_regresion_rroc_aoc(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    return measure_rroc_aoc(
        synthetic_x,
        synthetic_y,
        reral_x,
        real_y,
        LinearRegression,
        n_jobs=-1,
    )


def measure_random_forest_rroc_aoc(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    return measure_rroc_aoc(
        synthetic_x,
        synthetic_y,
        reral_x,
        real_y,
        RandomForestRegressor,
    )


def measure_tabpfn_rroc_aoc(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    return measure_rroc_aoc(
        synthetic_x,
        synthetic_y,
        reral_x,
        real_y,
        TabPFNRegressor,
        n_estimators=len(reral_x.columns) * 2,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )


def measure_xgb_rroc_aoc(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    n_estimators = len(reral_x.columns) * 2
    max_depth = max(len(reral_x.columns) // 5, 4)

    return measure_rroc_aoc(
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
        (0, measure_random_forest_rroc_aoc([synth_x], [synth_y], test_x, test_y)[0])
    )


def xgb_process(
    downstream_results_queue: Queue,
    synth_x: pd.DataFrame,
    synth_y: pd.DataFrame,
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
):
    downstream_results_queue.put(
        (1, measure_xgb_rroc_aoc([synth_x], [synth_y], test_x, test_y)[0])
    )


def tabpfn_process(
    downstream_results_queue: Queue,
    synth_x: pd.DataFrame,
    synth_y: pd.DataFrame,
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
):
    downstream_results_queue.put(
        (3, measure_tabpfn_rroc_aoc([synth_x], [synth_y], test_x, test_y)[0])
    )


def logistic_regression_process(
    downstream_results_queue: Queue,
    synth_x: pd.DataFrame,
    synth_y: pd.DataFrame,
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
):
    downstream_results_queue.put(
        (2, measure_linear_regresion_rroc_aoc([synth_x], [synth_y], test_x, test_y)[0])
    )


def calculate_rroc_aoc(
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
            name="RandomForestRegressor",
        )
    )
    downstream_jobs.append(
        Process(
            target=logistic_regression_process,
            args=(downstream_results_queue, synth_x, synth_y, real_x, real_y),
            name="LinearRegression",
        )
    )
    downstream_jobs.append(
        Process(
            target=xgb_process,
            args=(downstream_results_queue, synth_x, synth_y, real_x, real_y),
            name="XGBRegressor",
        )
    )
    downstream_jobs.append(
        Process(
            target=tabpfn_process,
            args=(downstream_results_queue, synth_x, synth_y, real_x, real_y),
            name="TabPFNRegressor",
        )
    )

    for job in downstream_jobs:
        job.start()
    for job in downstream_jobs:
        logger.debug(f"Waiting for {job.name} to finish...")
        job.join()
    return [
        accuracy
        for _, accuracy in sorted(
            downstream_results_queue.get() for _ in range(len(downstream_jobs))
        )
    ]
