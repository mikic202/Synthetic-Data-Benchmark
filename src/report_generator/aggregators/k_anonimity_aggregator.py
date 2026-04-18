from pathlib import Path
import glob
from typing import Any
import json
import pandas as pd


class K_anonimity_aggregator:
    def __init__(
        self, input_paths: list[Path], output_path: Path, generator_types: list[str]
    ) -> None:
        self._k_aninimity_paths = [
            k_aninimity_file
            for path in input_paths
            for k_aninimity_file in glob.glob(
                str(path / "**/k-anonimity.json"), recursive=True
            )
        ]
        self._generator_types = generator_types
        self._output_path = output_path

    def __call__(self) -> Any:
        k_anonimity_results = {}
        for generator_type, k_aninimity_path in zip(
            self._generator_types, self._k_aninimity_paths
        ):
            with open(k_aninimity_path) as k_aninimity_file:
                k_anonimity_data = json.load(k_aninimity_file)
            k_anonimity_results[generator_type] = {
                **k_anonimity_data["cl_dataset_avg"],
                **k_anonimity_data["reg_dataset_avg"],
            }
        k_anonimity_dataframe = pd.DataFrame(k_anonimity_results)
        k_anonimity_dataframe.to_latex(self._output_path / "k-anonimity.tex")
