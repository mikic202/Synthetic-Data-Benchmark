from pathlib import Path
import glob
from typing import Any
import json
import pandas as pd
from src.benchmark_tool.benchmark_tool import (
    AVAILABLE_CLASSIFICATION_DATASETS,
    AVALIABLE_REGRESSION_DATASETS,
)
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from src.test_datasets.datset_utils import names_in_report, map_dataset_name_to_name_in_report

dataset_names = list(AVAILABLE_CLASSIFICATION_DATASETS.keys()) + list(
    AVALIABLE_REGRESSION_DATASETS.keys()
)

class SingleValueAggregator:
    def __init__(
        self,
        input_paths: list[Path],
        output_path: Path,
        generator_types: list[str],
        filename: str,
        metric_name: str = None,
        refernce_generator_type: str = "real",
    ) -> None:
        if metric_name:
            self._metric_name = metric_name
        else:
            self._metric_name = filename
        self._file_paths = []
        for path, is_metric_generated in zip(input_paths, generator_types):
            for found_file in glob.glob(
                str(path / f"**/{filename}.json"), recursive=True
            ):
                if is_metric_generated:
                    if not refernce_generator_type or refernce_generator_type not in is_metric_generated:
                        self._file_paths.append(found_file)
                    else:
                        self._refernce_generator_filename = found_file
        self._generator_types = generator_types
        self._refernce_generator_type = refernce_generator_type
        self._output_path = output_path
        self._filename = filename

    def __call__(self) -> Any:
        results = {}
        result_stds = {}
        results_p_values = {}
        for generator_type, k_aninimity_path in zip(
            self._generator_types, self._file_paths
        ):
            with open(k_aninimity_path) as k_aninimity_file:
                data = json.load(k_aninimity_file)
            results[generator_type] = {
                **data["cl_dataset_avg"],
                **data["reg_dataset_avg"],
            }
            result_stds[generator_type] = {
                **data["cl_dataset_std"],
                **data["reg_dataset_std"],
            }
            results_p_values[generator_type] = {**data["cl_p_value"],
                **data["reg_p_value"]}

        dataframe = pd.DataFrame(results).reindex(dataset_names)

        if self._refernce_generator_type:
            with open(self._refernce_generator_filename) as k_aninimity_file:
                data = json.load(k_aninimity_file)
            reference_results = {
                **data["cl_dataset_avg"],
                **data["reg_dataset_avg"],
            }
            result_stds[self._refernce_generator_type]  = {
                **data["cl_dataset_std"],
                **data["reg_dataset_std"],
            }
            results_p_values[self._refernce_generator_type] = {**data["cl_p_value"],
                **data["reg_p_value"]}
            referance_results_df = pd.Series(reference_results).reindex(dataset_names)
            wilconox_values = {self._refernce_generator_type: ""}
            for generator in dataframe:
                wilconox_values[generator] = wilcoxon(referance_results_df, dataframe[generator]).pvalue
            dataframe[self._refernce_generator_type] = referance_results_df
        p_values_dataframe = pd.DataFrame(results_p_values).reindex(dataset_names)
        plt.figure(figsize=(14, 6))
        flipped_matrix = dataframe.rename(index=names_in_report).T
        for generator, row in flipped_matrix.iterrows():
            plt.scatter(
                flipped_matrix.columns,
                row,
                label=generator,
                s=60,
                alpha=0.8,
            )
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.ylabel(self._metric_name)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(self._output_path / f"{self._filename}.png", dpi=300)
        plt.close()
        std_dataframe = pd.DataFrame(result_stds).reindex(dataset_names)
        dataframe = (
            dataframe.map("${:.2f}".format)
            + " \pm "
            + std_dataframe.map("{:.2f} \\atop".format) + p_values_dataframe.map("{:.2f}$".format)
        )
        dataframe = dataframe.rename(index=names_in_report)
        cl_df = dataframe.loc[[idx for idx in map(map_dataset_name_to_name_in_report, list(AVAILABLE_CLASSIFICATION_DATASETS.keys())) if idx in dataframe.index]]
        reg_df = dataframe.loc[[idx for idx in map(map_dataset_name_to_name_in_report, list(AVALIABLE_REGRESSION_DATASETS.keys())) if idx in dataframe.index]]
        combined = pd.concat(
            {"Zadania klasyfikacji": cl_df, "Zadania regresji": reg_df, "Wilcoxon": pd.DataFrame({"": wilconox_values}).T} if self._refernce_generator_type else {"Zadania klasyfikacji": cl_df, "Zadania regresji": reg_df}
        )
        combined.to_latex(
            self._output_path / f"{self._filename}.tex",
            escape=False,
            multicolumn=True,
            multicolumn_format="l",
            column_format="ll" + "c" * len(combined.columns),
            float_format="%.3f"
        )