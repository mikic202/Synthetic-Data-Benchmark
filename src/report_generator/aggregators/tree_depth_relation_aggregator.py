from pathlib import Path
import glob
from typing import Any
import json
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


class TreeDepthRelationAggregator:
    def __init__(
        self,
        input_paths: list[Path],
        output_path: Path,
        generator_types: list[str],
    ) -> None:
        self._filename = "tree_depth_precision_relation"
        self._file_paths = [
            found_file
            for path in input_paths
            for found_file in glob.glob(
                str(path / f"**/{self._filename}.json"), recursive=True
            )
        ]
        self._generator_types = generator_types
        self._output_path = output_path

    def __call__(self) -> Any:
        clasification_results = defaultdict(dict)
        regression_results = defaultdict(dict)
        for generator_type, k_aninimity_path in zip(
            self._generator_types, self._file_paths
        ):
            with open(k_aninimity_path) as k_aninimity_file:
                data = json.load(k_aninimity_file)
            for dataset, values in data["clasification"].items():
                clasification_results[dataset][generator_type] = values[0]
            for dataset, values in data["regression"].items():
                regression_results[dataset][generator_type] = values[0]

        for dataset in clasification_results:
            data = pd.DataFrame(clasification_results[dataset])
            data.index = data.index + 1
            plt.figure()
            plt.plot(data)
            plt.legend(data.columns)
            plt.title(f"Tree depth accuracy realtion for \n the {dataset} dataset")
            plt.xlabel("Tree Depth")
            plt.ylabel("Accuracy")
            plt.savefig(self._output_path / dataset)
            plt.close()

        for dataset in regression_results:
            data = pd.DataFrame(regression_results[dataset])
            data.index = data.index + 1
            plt.figure()
            plt.plot(data)
            plt.legend(data.columns)
            plt.title(f"Tree depth $R^2$ realtion for \n the {dataset} dataset")
            plt.xlabel("Tree Depth")
            plt.ylabel("$R^2$")
            plt.savefig(self._output_path / dataset)
            plt.close()

        # dataframe.to_latex(self._output_path / f"{self._filename}.tex")
