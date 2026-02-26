import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from tabicl import TabICLClassifier
from sklearn.linear_model import LogisticRegression
import torch
from sklearn.preprocessing import StandardScaler
from multiprocessing import Process, Queue
import multiprocessing as mp
from logging import getLogger


logger = getLogger(__name__)


NUMBER_OF_UNIQUE_ELEMENTS_FOR_BINARY_CLASIFICATION = 2


def measure_auc(
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
        clasifier = model_class(**kwargs).fit(scalled_synth_x, synth_y)
        if len(np.unique(real_y)) > NUMBER_OF_UNIQUE_ELEMENTS_FOR_BINARY_CLASIFICATION:
            areas_under_curve.append(
                roc_auc_score(
                    real_y,
                    clasifier.predict_proba(scalled_real_x),
                    multi_class="ovr",
                )
            )
            continue
        areas_under_curve.append(
            (
                roc_auc_score(real_y, clasifier.predict_proba(scalled_real_x)[:, 1]),
                clasifier.score(scalled_real_x, real_y),
            )
        )
    return areas_under_curve


def measure_logistic_regresion_auc(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    return measure_auc(
        synthetic_x,
        synthetic_y,
        reral_x,
        real_y,
        LogisticRegression,
        solver="saga",
        penalty="l2",
        C=1.0,
        tol=1e-3,
        max_iter=500,
        n_jobs=-1,
        multi_class="auto",
    )


def measure_random_forest_auc(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    return measure_auc(
        synthetic_x,
        synthetic_y,
        reral_x,
        real_y,
        RandomForestClassifier,
    )


def measure_tabpfn_auc(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    return measure_auc(
        synthetic_x,
        synthetic_y,
        reral_x,
        real_y,
        TabPFNClassifier,
        n_estimators=len(reral_x.columns) * 2,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )


def measure_tabicl_auc(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    return measure_auc(
        synthetic_x,
        synthetic_y,
        reral_x,
        real_y,
        TabICLClassifier,
        n_estimators=len(reral_x.columns) * 2,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )


def measure_xgb_auc(
    synthetic_x: list[pd.DataFrame],
    synthetic_y: list[list[int]],
    reral_x: pd.DataFrame,
    real_y: list[int],
):
    n_estimators = len(reral_x.columns) * 2
    max_depth = max(len(reral_x.columns) // 5, 4)

    return measure_auc(
        synthetic_x,
        synthetic_y,
        reral_x,
        real_y,
        XGBClassifier,
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
        (0, measure_random_forest_auc([synth_x], [synth_y], test_x, test_y)[0])
    )


def xgb_process(
    downstream_results_queue: Queue,
    synth_x: pd.DataFrame,
    synth_y: pd.DataFrame,
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
):
    downstream_results_queue.put(
        (1, measure_xgb_auc([synth_x], [synth_y], test_x, test_y)[0])
    )


def tabpfn_process(
    downstream_results_queue: Queue,
    synth_x: pd.DataFrame,
    synth_y: pd.DataFrame,
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
):
    downstream_results_queue.put(
        (3, measure_tabpfn_auc([synth_x], [synth_y], test_x, test_y)[0])
    )


def logistic_regression_process(
    downstream_results_queue: Queue,
    synth_x: pd.DataFrame,
    synth_y: pd.DataFrame,
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
):
    downstream_results_queue.put(
        (2, measure_logistic_regresion_auc([synth_x], [synth_y], test_x, test_y)[0])
    )


def tabicl_process(
    downstream_results_queue: Queue,
    synth_x: pd.DataFrame,
    synth_y: pd.DataFrame,
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
):
    downstream_results_queue.put(
        (4, measure_tabicl_auc([synth_x], [synth_y], test_x, test_y)[0])
    )


def calculate_auc(
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
    return [
        accuracy
        for _, accuracy in sorted(
            downstream_results_queue.get() for _ in range(len(downstream_jobs))
        )
    ]
