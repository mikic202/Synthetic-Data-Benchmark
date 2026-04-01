import argparse
from pathlib import Path
from src.report_generator.postprocessors.k_anonimity_postprocesssor import (
    KAnonimityPostprocessor,
)
from src.report_generator.postprocessors.base_postprocessor import RawData


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the postprocessed benchmark results should be saved",
    )

    parser.add_argument(
        "input_path",
        type=Path,
        help="Directory where the benchmark results are saved",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if not args.output_dir:
        args.output_dir = args.input_path / "processed"
    args.output_dir.mkdir(exist_ok=True, parents=True)
    k_anonimity_postprocessor = KAnonimityPostprocessor(args.output_dir)
    raw_data = RawData(args.input_path)
    k_anonimity_postprocessor(raw_data)
