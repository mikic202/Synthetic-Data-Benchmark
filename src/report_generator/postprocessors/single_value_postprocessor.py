from pathlib import Path

import scipy.stats
from src.report_generator.postprocessors.base_postprocessor import (
    BasePostprocessor,
    RawData,
)
import json
import numpy as np
import scipy


class SingleValuePostprocessor(BasePostprocessor):
    def __init__(
        self, output_path: Path, postprocessing_index: str, output_filename: str
    ) -> None:
        self._output_path = output_path / f"{output_filename}.json"
        self._postprocessing_index = postprocessing_index

    def __call__(self, raw_data: RawData) -> None:
        averages_per_clasification_dataset = {}
        stds_per_clasification_dataset = {}
        p_values_for_clasification = {}
        for dataset_name, dataset_results in raw_data.clasification_results[
            self._postprocessing_index
        ].items():
            dataset_results = np.array(
                dataset_results, dtype=float
            )
            averages_per_clasification_dataset[dataset_name] = dataset_results.mean()
            stds_per_clasification_dataset[dataset_name] = dataset_results.std()
            p_values_for_clasification[dataset_name] = scipy.stats.shapiro(dataset_results).pvalue

        averages_per_regression_dataset = {}
        stds_per_regression_dataset = {}
        p_values_for_regression = {}

        for dataset_name, dataset_results in raw_data.regression_results[
            self._postprocessing_index
        ].items():
            dataset_results = np.array(
                dataset_results, dtype=float
            )
            averages_per_regression_dataset[dataset_name] = dataset_results.mean()
            stds_per_regression_dataset[dataset_name] = dataset_results.std()
            p_values_for_regression[dataset_name] = scipy.stats.shapiro(dataset_results).pvalue


        with open(self._output_path, "w") as processed_result_files:
            json.dump(
                {
                    "cl_dataset_avg": averages_per_clasification_dataset,
                    "reg_dataset_avg": averages_per_regression_dataset,
                    "cl_dataset_std": stds_per_clasification_dataset,
                    "reg_dataset_std": stds_per_regression_dataset,
                    "cl_p_value": p_values_for_clasification,
                    "reg_p_value": p_values_for_regression,
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
