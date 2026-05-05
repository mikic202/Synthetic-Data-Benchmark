from pathlib import Path
import glob
import json
import pandas as pd


class SingleValueAggregator:
    def __init__(
        self,
        input_paths: list[Path],
        reference_data_path: Path,
        output_path: Path,
        generator_types: list[str],
        filename: str,
    ) -> None:
        self._file_paths = [
            found_file
            for path in input_paths
            for found_file in glob.glob(
                str(path / f"**/{filename}.json"), recursive=True
            )
        ]
        self._reference_path = list(
            glob.glob(str(reference_data_path / f"**/{filename}.json"), recursive=True)
        )[0]
        self._generator_types = generator_types
        self._output_path = output_path
        self._filename = filename

    def __call__(self) -> None:
        results = {}
        for generator_type, data_path in zip(
            [*self._generator_types, "reference"],
            [*self._file_paths, self._reference_path],
        ):
            with open(data_path) as data_file:
                data = json.load(data_file)
            results[generator_type] = {
                **data["cl_dataset_avg"],
                **data["reg_dataset_avg"],
            }
        dataframe = pd.DataFrame(results)
        dataframe.loc["average"] = dataframe.mean()
        dataframe.to_latex(
            self._output_path / f"{self._filename}.tex",
            column_format="l" + "c" * (dataframe.shape[1] - 1) + "|" + "c",
        )
