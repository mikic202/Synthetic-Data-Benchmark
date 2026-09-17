from pathlib import Path
import glob
from typing import Any
import json
import pandas as pd
import matplotlib.pyplot as plt


class AucAggregator:
    def __init__(
        self,
        input_paths: list[Path],
        output_path: Path,
        generator_types: list[str],
    ) -> None:
        self._file_paths = [
            found_file
            for path, is_metric_generated in zip(input_paths, generator_types)
            for found_file in glob.glob(str(path / "**/auc.json"), recursive=True)
            if is_metric_generated
        ]
        self._generator_types = generator_types
        self._output_path = output_path

    def __call__(self) -> Any:
        clasification_results = {}
        clasification_stds = {}
        regression_results = {}
        regression_stds = {}
        for generator_type, k_aninimity_path in zip(
            self._generator_types, self._file_paths
        ):
            with open(k_aninimity_path) as k_aninimity_file:
                data = json.load(k_aninimity_file)
            clasification_stds[generator_type] = data["clasification_std"]
            clasification_results[generator_type] = data["clasification_avg"]
            regression_stds[generator_type] = data["regression_std"]
            regression_results[generator_type] = data["regression_avg"]
        clasification_results = pd.DataFrame(clasification_results)

        plt.figure(figsize=(14, 6))
        for generator, row in clasification_results.T.iterrows():
            plt.scatter(
                clasification_results.T.columns,
                row,
                label=generator,
                s=60,
                alpha=0.8,
            )
        plt.ylabel("Pole pod krzywą ROC")
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.tight_layout()
        plt.savefig(self._output_path / f"auc_clasification.png", dpi=300)
        plt.close()

        clasification_stds = pd.DataFrame(clasification_stds)
        (
            clasification_results.map("${:.2f}".format)
            + " \pm "
            + clasification_stds.map("{:.2f}$".format)
        ).to_latex(self._output_path / "auc_clasification.tex", column_format="l" + "c" * len(clasification_results.columns))

        regression_results = pd.DataFrame(regression_results)
        regression_stds = pd.DataFrame(regression_stds)

        plt.figure(figsize=(14, 6))
        for generator, row in regression_results.T.iterrows():
            plt.scatter(
                regression_results.T.columns,
                row,
                label=generator,
                s=60,
                alpha=0.8,
            )
        plt.ylabel("C-index")
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.tight_layout()
        plt.savefig(self._output_path / f"auc_regression.png", dpi=300)
        plt.close()

        (
            regression_results.map("${:.2f}".format)
            + " \pm "
            + regression_stds.map("{:.2f}$".format)
        ).to_latex(
            self._output_path / "auc_regression.tex", column_format="l" + "c" * len(regression_results.columns)
        )


