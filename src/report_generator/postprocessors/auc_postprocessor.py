from pathlib import Path
from src.report_generator.postprocessors.base_postprocessor import (
    BasePostprocessor,
    RawData,
)
import json
import pandas as pd


class AUCPostprocessor(BasePostprocessor):
    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path / "auc.json"

    def __call__(self, raw_data: RawData) -> None:

        results_for_clasification = []
        results_for_clasification_for_given_dataset = {}
        results_for_regression = []
        results_for_regression_for_given_dataset = {}
        for dataset_name, dataset_results in raw_data.clasification_results[
            "area-under-curve"
        ].items():
            dataset_values = [
                    pd.DataFrame(single_results).iloc[0]
                    for single_results in dataset_results
                ]
            results_for_clasification_for_given_dataset[dataset_name] = pd.concat(dataset_values, axis=1).T.astype(float).mean().to_dict()
            results_for_clasification.extend(
                dataset_values
            )

        for dataset_name, dataset_results in raw_data.regression_results["area-under-curve"].items():
            dataset_values = [
                    pd.DataFrame(single_results).iloc[0]
                    for single_results in dataset_results
                ]
            results_for_regression_for_given_dataset[dataset_name] = pd.concat(dataset_values, axis=1).T.astype(float).mean().to_dict()
            results_for_regression.extend(
                dataset_values
            )

        combined_clasification = pd.concat(results_for_clasification, axis=1).T.astype(float)
        combined_regression = pd.concat(results_for_regression, axis=1).T.astype(float)
        combined_both = pd.concat(
            results_for_clasification + results_for_regression, axis=1
        ).T.astype(float)
        with open(self._output_path, "w") as processed_result_files:
            json.dump(
                {
                    "clasification_avg": combined_clasification.mean().to_dict(),
                    "regression_avg": combined_regression.mean().to_dict(),
                    "clasification_std": combined_clasification.std().to_dict(),
                    "regression_std": combined_regression.std().to_dict(),
                    "avg": combined_both.mean().to_dict(),
                    "std": combined_both.std().to_dict(),
                    "per_dataset_means": results_for_clasification_for_given_dataset | results_for_regression_for_given_dataset
                },
                processed_result_files,
                indent=4,
            )
