from pathlib import Path
from src.report_generator.aggregators.single_value_aggregator import (
    SingleValueAggregator,
)
from src.constants import DISTANCE_TO_NEAREST_NEIGHBOUR
import json
import pandas as pd


class DistanceToNearestNeigbourAggregator(SingleValueAggregator):
    def __init__(
        self,
        input_paths: list[Path],
        reference_data_path: Path,
        output_path: Path,
        generator_types: list[str],
    ) -> None:
        super().__init__(
            input_paths,
            reference_data_path,
            output_path,
            generator_types,
            DISTANCE_TO_NEAREST_NEIGHBOUR,
        )

    def __call__(self) -> None:
        results = {}
        for generator_type, data_path in zip(
            self._generator_types,
            self._file_paths,
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
        )
