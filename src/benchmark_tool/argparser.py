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
    parser.add_argument(
        "-k", "--k-anonimity", action="store_true", help="Compute k-anonimity metric"
    )
    parser.add_argument(
        "-u",
        "--unlinkability",
        action="store_true",
        help="Compute unlinkability metric",
    )
    parser.add_argument(
        "-d",
        "--distance-to-nearest",
        action="store_true",
        help="Compute distance to nearest neighbour metric",
    )
    parser.add_argument(
        "-s",
        "--dataset-statistics",
        action="store_true",
        help="Compute dataset statistics",
    )
    parser.add_argument(
        "-t",
        "--tree-depth-precision-relation",
        action="store_true",
        help="Compute relation between tree depth and precision",
    )
    parser.add_argument(
        "-a",
        "--area-under-curve",
        action="store_true",
        help="Compute the area under curve metric",
    )
    parser.add_argument(
        "-c",
        "--convex-hull",
        action="store_true",
        help="Compute convex hull coverage",
    )
    parser.add_argument(
        "-ds",
        "--svn-discrimination",
        action="store_true",
        help="Compute how well svn discriminates examples",
    )

    return parser.parse_args()
