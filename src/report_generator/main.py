import argparse
from pathlib import Path
from src.report_generator.postprocessors.k_anonimity_postprocesssor import (
    KAnonimityPostprocessor,
)
from src.report_generator.postprocessors.distance_to_nearest_neighbour_postprocessor import (
    DistanceToNearestNeighbourPostprocessor,
)
from src.report_generator.postprocessors.unlinkability_postprocessor import (
    UnlinkabilityPostprocessor,
)
from src.report_generator.postprocessors.base_postprocessor import RawData
import json


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


def get_postprocessing_config(data_dir: Path, output_path: Path):
    run_param_file = data_dir / "run_params.json"
    if not run_param_file.exists():
        raise Exception("There is no file with a run params")
    with open(run_param_file, "r") as run_params_filehandle:
        run_params = json.load(run_params_filehandle)

    postprocessors = []
    if run_params["k_anonimity"] == "True":
        postprocessors.append(KAnonimityPostprocessor(output_path))
    if run_params["distance_to_nearest"] == "True":
        postprocessors.append(DistanceToNearestNeighbourPostprocessor(output_path))
    if run_params["unlinkability"] == "True":
        postprocessors.append(UnlinkabilityPostprocessor(output_path))

    return postprocessors


def main():
    args = parse_args()
    if not args.output_dir:
        args.output_dir = args.input_path / "processed"
    args.output_dir.mkdir(exist_ok=True, parents=True)
    postprocessors = get_postprocessing_config(args.input_path, args.output_dir)
    raw_data = RawData(args.input_path)
    for postprocessor in postprocessors:
        postprocessor(raw_data)
