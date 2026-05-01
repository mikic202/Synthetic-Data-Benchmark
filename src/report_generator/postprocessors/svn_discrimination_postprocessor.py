from pathlib import Path
from src.report_generator.postprocessors.single_value_postprocessor import (
    SingleValuePostprocessor,
)
from src.constants import DISCRIMINATION


class SvnDiscriminationPostprocessor(SingleValuePostprocessor):
    def __init__(self, output_path: Path) -> None:
        super().__init__(output_path, DISCRIMINATION, DISCRIMINATION)
