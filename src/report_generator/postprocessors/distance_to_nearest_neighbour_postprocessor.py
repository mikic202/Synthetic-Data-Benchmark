from pathlib import Path
from src.report_generator.postprocessors.base_postprocessor import (
    BasePostprocessor,
    RawData,
)
import json
from src.constants import DISTANCE_TO_NEAREST_NEIGHBOUR
import numpy as np


class DistanceToNearestNeighbourPostprocessor(BasePostprocessor):
    def __init__(self, output_path: Path, data_key = DISTANCE_TO_NEAREST_NEIGHBOUR) -> None:
        self._output_path = output_path
        self._data_key = data_key

    def __call__(self, raw_data: RawData) -> None:
        averages_per_clasification_dataset = {}
        stds_per_clasification_dataset = {}
        for dataset_name, dataset_results in raw_data.clasification_results[
            self._data_key
        ].items():
            averages_per_clasification_dataset[dataset_name] = np.mean(
                [float(result["mean"]) for result in dataset_results]
            )
            stds_per_clasification_dataset[dataset_name] = np.std(
                [float(result["mean"]) for result in dataset_results]
            )

        averages_per_regression_dataset = {}
        stds_per_regression_dataset = {}

        for dataset_name, dataset_results in raw_data.regression_results[
            self._data_key
        ].items():
            averages_per_regression_dataset[dataset_name] = np.mean(
                [float(result["mean"]) for result in dataset_results]
            )
            stds_per_regression_dataset[dataset_name] = np.std(
                [float(result["mean"]) for result in dataset_results]
            )

        with open(
            self._output_path / f"{self._data_key}.json", "w"
        ) as processed_result_files:
            json.dump(
                {
                    "cl_dataset_avg": averages_per_clasification_dataset,
                    "reg_dataset_avg": averages_per_regression_dataset,
                    "cl_dataset_std": stds_per_clasification_dataset,
                    "reg_dataset_std": stds_per_regression_dataset,
                    "clasification_avg": np.mean(
                        list(averages_per_clasification_dataset.values())
                    ),
                    "regression_avg": np.mean(
                        list(averages_per_regression_dataset.values())
                    ),
                    "avg": np.mean(
                        list({
                            **averages_per_clasification_dataset,
                            **averages_per_regression_dataset,
                        }.values())
                    ),
                    "std": np.std(
                        list({
                            **averages_per_clasification_dataset,
                            **averages_per_regression_dataset,
                        }.values())
                    ),
                    "clasification_std": np.std(
                        list(averages_per_clasification_dataset.values())
                    ),
                    "regression_std": np.std(
                        list(averages_per_regression_dataset.values())
                    ),
                },
                processed_result_files,
                indent=4,
            )
