from pathlib import Path
from src.report_generator.postprocessors.base_postprocessor import (
    BasePostprocessor,
    RawData,
)
import json
import numpy as np
import matplotlib.pyplot as plt


class TreeDepthPostprocessor(BasePostprocessor):
    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path

    def __call__(self, raw_data: RawData) -> None:
        averages_per_clasification_dataset = {
            dataset_name: (
                np.array([list(result.values()) for result in dataset_results])
                .mean(axis=0)
                .tolist(),
                list(dataset_results[0].keys()),
            )
            for dataset_name, dataset_results in raw_data.clasification_results[
                "tree-depth-precision-relation"
            ].items()
        }
        averages_per_regression_dataset = {
            dataset_name: (
                np.array([list(result.values()) for result in dataset_results])
                .mean(axis=0)
                .tolist(),
                list(dataset_results[0].keys()),
            )
            for dataset_name, dataset_results in raw_data.regression_results[
                "tree-depth-precision-relation"
            ].items()
        }
        with open(
            self._output_path / "tree_depth_precision_relation.json", "w"
        ) as output_file:
            json.dump(
                {
                    "clasification": averages_per_clasification_dataset,
                    "regression": averages_per_regression_dataset,
                },
                output_file,
                indent=4,
            )
        plot_folder = self._output_path / "tree_depth_precision_relation"
        plot_folder.mkdir(exist_ok=True)
        for dataset in averages_per_clasification_dataset:
            plt.figure()
            plt.ylim((0, 1))
            plt.plot(
                averages_per_clasification_dataset[dataset][1],
                averages_per_clasification_dataset[dataset][0],
            )
            plt.savefig(plot_folder / dataset)

        for dataset in averages_per_regression_dataset:
            plt.figure()
            plt.plot(
                averages_per_regression_dataset[dataset][1],
                averages_per_regression_dataset[dataset][0],
            )
            plt.savefig(plot_folder / dataset)
