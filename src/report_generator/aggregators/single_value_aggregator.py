from pathlib import Path
import glob
from typing import Any
import json
import pandas as pd


class SingleValueAggregator:
    def __init__(
        self,
        input_paths: list[Path],
        output_path: Path,
        generator_types: list[str],
        filename: str,
    ) -> None:
        self._file_paths = [
            found_file
            for path, is_metric_generated in zip(input_paths, generator_types)
            for found_file in glob.glob(
                str(path / f"**/{filename}.json"), recursive=True
            )
            if is_metric_generated
        ]
        self._generator_types = generator_types
        self._output_path = output_path
        self._filename = filename

    def __call__(self) -> Any:
        results = {}
        for generator_type, k_aninimity_path in zip(
            self._generator_types, self._file_paths
        ):
            with open(k_aninimity_path) as k_aninimity_file:
                data = json.load(k_aninimity_file)
            results[generator_type] = {
                **data["cl_dataset_avg"],
                **data["reg_dataset_avg"],
            }
        dataframe = pd.DataFrame(results)
        dataframe.to_latex(self._output_path / f"{self._filename}.tex")
