from pathlib import Path
import glob
from typing import Any
import json
import pandas as pd


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
        regression_results = {}
        for generator_type, k_aninimity_path in zip(
            self._generator_types, self._file_paths
        ):
            with open(k_aninimity_path) as k_aninimity_file:
                data = json.load(k_aninimity_file)
            clasification_results[generator_type] = data["clasification_avg"]
            regression_results[generator_type] = data["regression_avg"]
        pd.DataFrame(clasification_results).to_latex(
            self._output_path / "auc_clasification.tex"
        )
        pd.DataFrame(regression_results).to_latex(
            self._output_path / "auc_regression.tex"
        )
