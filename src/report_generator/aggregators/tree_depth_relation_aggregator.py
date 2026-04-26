from pathlib import Path
import glob
from typing import Any
import json
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


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
        self._output_path = output_path / "tree_depth_relation"
        self._output_path.mkdir(exist_ok=True)

    def _save_cosine_similarity(
        self, graph_course: pd.DataFrame, output_location: Path
    ):
        cosine_dist_matrix = cosine_similarity(graph_course.T)
        cosine_sim_df = pd.DataFrame(
            cosine_dist_matrix, index=graph_course.columns, columns=graph_course.columns
        )
        cosine_sim_df.to_latex(output_location)

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
            self._save_cosine_similarity(data, self._output_path / (dataset + ".tex"))

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
            self._save_cosine_similarity(data, self._output_path / (dataset + ".tex"))
        # dataframe.to_latex(self._output_path / f"{self._filename}.tex")
