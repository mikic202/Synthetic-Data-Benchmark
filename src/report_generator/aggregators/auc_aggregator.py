from pathlib import Path
import glob
from typing import Any
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


class AucAggregator:
    def __init__(
        self,
        input_paths: list[Path],
        output_path: Path,
        generator_types: list[str],
        refernce_generator_type: str = "real",
    ) -> None:
        self._file_paths = []
        for path, is_metric_generated in zip(input_paths, generator_types):
            for found_file in glob.glob(
                str(path / "**/auc.json"), recursive=True
            ):
                if is_metric_generated:
                    if refernce_generator_type not in is_metric_generated:
                        self._file_paths.append(found_file)
                    else:
                        self._refernce_generator_filename = found_file
        self._generator_types = generator_types
        self._output_path = output_path
        self._refernce_generator_type = refernce_generator_type

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

        clasification_reference_values = {}
        regression_reference_values = {}

        with open(self._refernce_generator_filename) as k_aninimity_file:
            data = json.load(k_aninimity_file)
        clasification_stds[self._refernce_generator_type] = data["clasification_std"]
        clasification_reference_values = data["clasification_avg"]
        regression_stds[self._refernce_generator_type] = data["regression_std"]
        regression_reference_values = data["regression_avg"]

        clasification_referance_results_df = pd.Series(clasification_reference_values)
        regression_referance_results_df = pd.Series(regression_reference_values)
        wilconox_values = {self._refernce_generator_type: ""}
        for generator in clasification_results:
            wilconox_values[generator] = wilcoxon(clasification_referance_results_df, clasification_results[generator]).pvalue

        clasification_results[self._refernce_generator_type] = clasification_referance_results_df


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
        clasification_results = (
            clasification_results.map("${:.2f}".format)
            + " \pm "
            + clasification_stds.map("{:.2f}$".format)
        )
        clasification_results.loc["Wilcoxon"] = pd.Series(wilconox_values).T
        clasification_results.to_latex(self._output_path / "auc_clasification.tex", column_format="l" + "c" * len(clasification_results.columns))
        regression_results = pd.DataFrame(regression_results)
        regression_stds = pd.DataFrame(regression_stds)

        wilconox_values = {self._refernce_generator_type: ""}
        for generator in regression_results:
            wilconox_values[generator] = wilcoxon(regression_referance_results_df, regression_results[generator]).pvalue

        regression_results[self._refernce_generator_type] = regression_referance_results_df

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

        regression_results = (
            regression_results.map("${:.2f}".format)
            + " \pm "
            + regression_stds.map("{:.2f}$".format)
        )
        regression_results.loc["Wilcoxon"] = pd.Series(wilconox_values).T
        regression_results.to_latex(
            self._output_path / "auc_regression.tex", column_format="l" + "c" * len(regression_results.columns)
        )


