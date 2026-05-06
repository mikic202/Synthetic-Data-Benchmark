from pathlib import Path
from src.report_generator.postprocessors.base_postprocessor import (
    BasePostprocessor,
    RawData,
)
import statistics
import json
import numpy as np


class SingleValuePostprocessor(BasePostprocessor):
    def __init__(
        self, output_path: Path, postprocessing_index: str, output_filename: str
    ) -> None:
        self._output_path = output_path / f"{output_filename}.json"
        self._postprocessing_index = postprocessing_index

    def __call__(self, raw_data: RawData) -> None:
        averages_per_clasification_dataset = {}
        stds_per_clasification_dataset = {}
        for dataset_name, dataset_results in raw_data.clasification_results[
            self._postprocessing_index
        ].items():
            averages_per_clasification_dataset[dataset_name] = np.array(
                dataset_results, dtype=float
            ).mean()
            stds_per_clasification_dataset[dataset_name] = np.array(
                dataset_results, dtype=float
            ).std()

        averages_per_regression_dataset = {}
        stds_per_regression_dataset = {}

        for dataset_name, dataset_results in raw_data.regression_results[
            self._postprocessing_index
        ].items():
            averages_per_regression_dataset[dataset_name] = np.array(
                dataset_results, dtype=float
            ).mean()
            stds_per_regression_dataset[dataset_name] = np.array(
                dataset_results, dtype=float
            ).std()

        with open(self._output_path, "w") as processed_result_files:
            json.dump(
                {
                    "cl_dataset_avg": averages_per_clasification_dataset,
                    "reg_dataset_avg": averages_per_regression_dataset,
                    "cl_dataset_std": stds_per_clasification_dataset,
                    "reg_dataset_std": stds_per_regression_dataset,
                    "clasification_avg": statistics.mean(
                        averages_per_clasification_dataset.values()
                    ),
                    "regression_avg": statistics.mean(
                        averages_per_regression_dataset.values()
                    ),
                    "avg": statistics.mean(
                        {
                            **averages_per_clasification_dataset,
                            **averages_per_regression_dataset,
                        }.values()
                    ),
                    "std": statistics.stdev(
                        {
                            **averages_per_clasification_dataset,
                            **averages_per_regression_dataset,
                        }.values()
                    ),
                    "clasification_std": statistics.stdev(
                        averages_per_clasification_dataset.values()
                    ),
                    "regression_std": statistics.stdev(
                        averages_per_regression_dataset.values()
                    ),
                },
                processed_result_files,
                indent=4,
            )
