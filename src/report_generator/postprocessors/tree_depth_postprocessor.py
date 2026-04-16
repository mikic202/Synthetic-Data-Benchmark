from pathlib import Path
from src.report_generator.postprocessors.base_postprocessor import (
    BasePostprocessor,
    RawData,
)
import statistics
import json
import numpy as np
import matplotlib.pyplot as plt


class TreeDepthPostprocessor(BasePostprocessor):
    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path

    def __call__(self, raw_data: RawData) -> None:
        averages_per_clasification_dataset = {
            dataset_name: (
                np.array(
                    [
                        list(result["tree-depth-precision-relation"].values())
                        for result in dataset_results
                    ]
                ).mean(axis=0),
                list(dataset_results[0]["tree-depth-precision-relation"].keys()),
            )
            for dataset_name, dataset_results in raw_data.clasification_results.items()
        }
        averages_per_regression_dataset = {
            dataset_name: (
                np.array(
                    [
                        list(result["tree-depth-precision-relation"].values())
                        for result in dataset_results
                    ]
                ).mean(axis=0),
                list(dataset_results[0]["tree-depth-precision-relation"].keys()),
            )
            for dataset_name, dataset_results in raw_data.regression_results.items()
        }
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
            plt.ylim((0, 1))
            plt.plot(
                averages_per_regression_dataset[dataset][1],
                averages_per_regression_dataset[dataset][0],
            )
            plt.savefig(plot_folder / dataset)
