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

    parser.add_argument(
        "-n",
        "--repetitions-number",
        type=int,
        default=5,
        help="Number of repetitions for each metric",
    )

    return parser.parse_args()
