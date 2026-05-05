from pathlib import Path
from src.report_generator.postprocessors.single_value_postprocessor import (
    SingleValuePostprocessor,
)
from src.constants import CONVEX_HULL


class ConvexHullPostprocessor(SingleValuePostprocessor):
    def __init__(self, output_path: Path) -> None:
        super().__init__(output_path, CONVEX_HULL, CONVEX_HULL)
