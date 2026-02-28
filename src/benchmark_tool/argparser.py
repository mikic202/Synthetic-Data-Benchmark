import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("./benchamr_results"),
        help="Directory where the benchmark results should be saved",
    )

    parser.add_argument(
        "generator_type",
        type=str,
        choices=["tabpfngen", "tabiclgen", "ctgan", "smote"],
        help="Choose which generator type needs to be benchamrked",
    )

    return parser.parse_args()
