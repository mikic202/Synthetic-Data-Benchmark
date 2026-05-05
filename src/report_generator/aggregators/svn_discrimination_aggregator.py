from pathlib import Path
from src.report_generator.aggregators.single_value_aggregator import (
    SingleValueAggregator,
)
from src.constants import DISCRIMINATION


class SvnDiscriminationAggregator(SingleValueAggregator):
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
            DISCRIMINATION,
        )
