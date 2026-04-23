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
from src.report_generator.postprocessors.tree_depth_postprocessor import (
    TreeDepthPostprocessor,
)
from src.report_generator.postprocessors.svn_discrimination_postprocessor import (
    SvnDiscriminationPostprocessor,
)
from src.report_generator.postprocessors.convex_hull_postprocessor import (
    ConvexHullPostprocessor,
)
from src.report_generator.aggregators.k_anonimity_aggregator import (
    KAnonimityAggregator,
)
from src.report_generator.aggregators.convex_hull_aggregator import ConvexHullAggregator
from src.report_generator.aggregators.unlinkability_aggregator import (
    UnlinkabilityAggregator,
)
from src.report_generator.aggregators.svn_discrimination_aggregator import (
    SvnDiscriminationAggregator,
)
from src.report_generator.aggregators.distance_to_nearest_aggregator import (
    DistanceToNearestNeigbourAggregator,
)
from src.report_generator.postprocessors.base_postprocessor import RawData
import json
from collections import defaultdict


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
        nargs="+",
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
    if run_params["tree_depth_precision_relation"] == "True":
        postprocessors.append(TreeDepthPostprocessor(output_path))
    if run_params["svn_discrimination"] == "True":
        postprocessors.append(SvnDiscriminationPostprocessor(output_path))
    if run_params["convex_hull"] == "True":
        postprocessors.append(ConvexHullPostprocessor(output_path))

    return postprocessors


def postprocess_single_run(input_path: Path, output_dir: Path | None):
    if not output_dir:
        output_dir = input_path / "processed"
    output_dir.mkdir(exist_ok=True, parents=True)
    postprocessors = get_postprocessing_config(input_path, output_dir)
    raw_data = RawData(input_path)
    for postprocessor in postprocessors:
        postprocessor(raw_data)


def get_tested_generators(input_paths: list[Path]):
    tested_generators = defaultdict(list)
    for input_dir in input_paths:
        run_param_file = input_dir / "run_params.json"
        if not run_param_file.exists():
            raise Exception("There is no file with a run params")
        with open(run_param_file, "r") as run_params_filehandle:
            run_params = json.load(run_params_filehandle)
        for metric in list(run_params.keys())[3:]:
            if run_params[metric] == "True":
                tested_generators[metric].append(run_params["generator_type"])
    return tested_generators


def main():
    args = parse_args()
    for input_dir in args.input_path:
        postprocess_single_run(input_dir, args.output_dir)

    combined_output_path = Path(".") / "combined_results"
    combined_output_path.mkdir(exist_ok=True)

    combined_path = Path("./combined_results")
    combined_path.mkdir(exist_ok=True)

    tested_generators = get_tested_generators(args.input_path)

    for metric in tested_generators:
        if metric == "k_anonimity":
            KAnonimityAggregator(
                args.input_path, combined_path, tested_generators[metric]
            )()
        elif metric == "convex_hull":
            ConvexHullAggregator(args.input_path, combined_path, tested_generators)()

        elif metric == "unlinkability":
            UnlinkabilityAggregator(args.input_path, combined_path, tested_generators)()
        elif metric == "svn_discrimination":
            SvnDiscriminationAggregator(
                args.input_path, combined_path, tested_generators
            )()
        elif metric == "distance_to_nearest":
            DistanceToNearestNeigbourAggregator(
                args.input_path, combined_path, tested_generators
            )()
