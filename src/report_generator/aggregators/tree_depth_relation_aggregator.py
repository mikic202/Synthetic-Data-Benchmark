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
        results = {}
        for generator_type, k_aninimity_path in zip(
            self._generator_types, self._file_paths
        ):
            with open(k_aninimity_path) as k_aninimity_file:
                data = json.load(k_aninimity_file)
            results[generator_type] = {
                **data["clasification"],
                # **data["reg_dataset_avg"],
            }
        results_grupped_by_dataset = defaultdict(dict)
        for generator_type in results:
            for dataset in results[generator_type]:
                results_grupped_by_dataset[dataset][generator_type] = results[
                    generator_type
                ][dataset][0]

        for dataset in results_grupped_by_dataset:
            data = pd.DataFrame(results_grupped_by_dataset[dataset])
            plt.figure()
            plt.plot(data)
            plt.legend(data.columns)
            plt.savefig(self._output_path / dataset)
            plt.close()

        # dataframe.to_latex(self._output_path / f"{self._filename}.tex")
