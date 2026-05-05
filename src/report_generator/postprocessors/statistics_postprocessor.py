from pathlib import Path
from src.report_generator.postprocessors.base_postprocessor import (
    BasePostprocessor,
    RawData,
)
import json
import pandas as pd


class StatisticsPostprocessor(BasePostprocessor):
    def __init__(self, output_path: Path) -> None:
        self._postprocessing_index = "dataset-statistics"
        self._output_path = output_path / f"{self._postprocessing_index}.json"

    def __call__(self, raw_data: RawData) -> None:
        averages_per_clasification_dataset = {
            dataset_name: pd.concat(
                [
                    each.stack()
                    for each in [pd.DataFrame(data) for data in dataset_results]
                ],
                axis=1,
            )
            .apply(lambda x: x.mean(), axis=1)
            .unstack()
            .to_dict()
            for dataset_name, dataset_results in raw_data.clasification_results[
                self._postprocessing_index
            ].items()
        }
        averages_per_regression_dataset = {
            dataset_name: pd.concat(
                [
                    each.stack()
                    for each in [pd.DataFrame(data) for data in dataset_results]
                ],
                axis=1,
            )
            .apply(lambda x: x.mean(), axis=1)
            .unstack()
            .to_dict()
            for dataset_name, dataset_results in raw_data.regression_results[
                self._postprocessing_index
            ].items()
        }

        with open(self._output_path, "w") as processed_result_files:
            json.dump(
                {
                    "cl_dataset_avg": averages_per_clasification_dataset,
                    "reg_dataset_avg": averages_per_regression_dataset,
                },
                processed_result_files,
                indent=4,
            )
