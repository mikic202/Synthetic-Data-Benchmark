from pathlib import Path
import glob
import numpy as np
import json
import pandas as pd


class DataseStatisticsAggregator:
    def __init__(
        self,
        input_paths: list[Path],
        reference_data_path: Path,
        output_path: Path,
        generator_types: list[str],
    ) -> None:
        self._filename = "dataset-statistics"
        self._file_paths = [
            found_file
            for path in input_paths
            for found_file in glob.glob(
                str(path / f"**/{self._filename}.json"), recursive=True
            )
        ]
        self._reference_path = list(
            glob.glob(
                str(reference_data_path / f"**/{self._filename}.json"), recursive=True
            )
        )[0]
        self._generator_types = generator_types
        self._output_path = output_path

    def __call__(self) -> None:
        reference_results = {}
        reference_mins = {}
        reference_ranges = {}
        with open(self._reference_path) as data_file:
            data = json.load(data_file)
        for dataset, dataset_statistics in {
            **data["cl_dataset_avg"],
            **data["reg_dataset_avg"],
        }.items():
            dataset_statistics_df = pd.DataFrame(dataset_statistics)

            essential_statistics = dataset_statistics_df.iloc[:10, :]
            reference_min = essential_statistics.min()
            reference_range = essential_statistics.max() - reference_min
            reference_range = reference_range.replace(0, 1)

            dataset_statistics_df.iloc[:10, :] = (
                essential_statistics - reference_min
            ) / reference_range
            reference_mins[dataset] = reference_min
            reference_ranges[dataset] = reference_range
            reference_results[dataset] = dataset_statistics_df

        results = {}

        def process_dataframe(df: pd.DataFrame, mins, range):
            df.iloc[:10, :] = (df.iloc[:10, :] - mins) / range
            return df

        for generator_type, data_path in zip(
            self._generator_types,
            self._file_paths,
        ):
            with open(data_path) as data_file:
                data = json.load(data_file)
            results[generator_type] = {
                dataset: np.linalg.norm(
                    process_dataframe(
                        pd.DataFrame(dataset_statistics),
                        reference_mins[dataset],
                        reference_ranges[dataset],
                    )
                    - reference_results[dataset]
                )
                for dataset, dataset_statistics in {
                    **data["cl_dataset_avg"],
                    **data["reg_dataset_avg"],
                }.items()
            }
        dataframe = pd.DataFrame(results)
        dataframe.loc["average"] = dataframe.mean()
        dataframe.to_latex(
            self._output_path / f"{self._filename}.tex",
        )
