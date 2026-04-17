from pathlib import Path
from src.report_generator.postprocessors.base_postprocessor import (
    BasePostprocessor,
    RawData,
)
import statistics
import json
import numpy as np


class ConvexHullPostprocessor(BasePostprocessor):
    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path

    def __call__(self, raw_data: RawData) -> None:
        averages_per_clasification_dataset = {
            dataset_name: np.array(dataset_results, dtype=float).mean()
            for dataset_name, dataset_results in raw_data.clasification_results[
                "convex-hull"
            ].items()
        }
        averages_per_regression_dataset = {
            dataset_name: np.array(dataset_results, dtype=float).mean()
            for dataset_name, dataset_results in raw_data.regression_results[
                "convex-hull"
            ].items()
        }

        with open(
            self._output_path / "convex-hull.json", "w"
        ) as processed_result_files:
            json.dump(
                {
                    "cl_dataset_avg": averages_per_clasification_dataset,
                    "reg_dataset_avg": averages_per_regression_dataset,
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
                },
                processed_result_files,
                indent=4,
            )
