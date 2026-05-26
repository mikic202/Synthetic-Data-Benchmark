from pathlib import Path
from src.report_generator.postprocessors.base_postprocessor import (
    BasePostprocessor,
    RawData,
)
import json
import pandas as pd


class AUCPostprocessor(BasePostprocessor):
    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path / f"auc.json"

    def __call__(self, raw_data: RawData) -> None:

        results_for_clasification = []
        results_for_regression = []
        for dataset_results in raw_data.clasification_results[
            "area-under-curve"
        ].values():
            results_for_clasification.extend(
                [
                    pd.DataFrame(single_results).iloc[0]
                    for single_results in dataset_results
                ]
            )

        for dataset_results in raw_data.regression_results["area-under-curve"].values():
            results_for_regression.extend(
                [
                    pd.DataFrame(single_results).iloc[0]
                    for single_results in dataset_results
                ]
            )

        combined_clasification = pd.concat(results_for_clasification, ignore_index=True)
        combined_regression = pd.concat(results_for_regression, ignore_index=True)
        combined_both = pd.concat(
            results_for_clasification + results_for_regression, ignore_index=True
        )

        with open(self._output_path, "w") as processed_result_files:
            json.dump(
                {
                    "clasification_avg": combined_clasification.mean().to_dict(),
                    "regression_avg": combined_regression.mean().to_dict(),
                    "clasification_std": combined_clasification.std().to_dict(),
                    "regression_std": combined_regression.std().to_dict(),
                    "avg": combined_both.mean().to_dict(),
                    "std": combined_both.std().to_dict(),
                },
                processed_result_files,
                indent=4,
            )
