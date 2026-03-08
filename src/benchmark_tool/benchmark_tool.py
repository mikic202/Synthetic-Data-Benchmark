from src.benchmark_tool.argparser import parse_args, save_args
from src.model_wrappers.full_tabpfn_gen import FullTabpfnGen
from src.model_wrappers.smote_generator import SmoteGenerator, SmoterGenerator
from src.model_wrappers.ctgan_generator import CTGANGenerator
from src.model_wrappers.neural_spline_flows_generator import NeuralSplineFlowsGenerator
from src.test_datasets import clasification_datasets, regression_datasets
from src.benchmark_tool import metric_wrappers
import pandas as pd
import datetime
from pathlib import Path
import json
from external.tab_pfn_gen.src.tabpfgen.tabpfgen import (
    TabPFGenRegressor,
    TabPFGenClassifier,
    TabICLClassifier,
)
from typing import Callable


AVAILABLE_CLASSIFICATION_DATASETS = {
    "mfeat_zernike": clasification_datasets.get_mfeat_zernike_dataset,
    "pc4": clasification_datasets.get_pc4_dataset,
    "climate_model_simulation": clasification_datasets.get_climate_model_simulation_dataset,
    "wdbc": clasification_datasets.get_wdbc_dataset,
    "analcatdata_authorship": clasification_datasets.get_analcatdata_authorship_dataset,
    "cardiovascular": clasification_datasets.get_cardiovascular_dataset,
}


AVALIABLE_REGRESSION_DATASETS = {
    "heart_failure_clinical_regresion": regression_datasets.get_heart_failure_clinical_regresion_dataset,
    "sleep_deprivation_and_cognitive_performance_regression": regression_datasets.get_sleep_deprivation_and_cognitive_performance_regression_dataset,
    "superconduct_regression": regression_datasets.get_superconduct_regression_dataset,
    "house_prices_regression": regression_datasets.get_house_prices_regression_dataset,
    "abalone": regression_datasets.get_abalone_dataset,
}


def get_clasification_model(args):
    match args.generator_type.lower():
        case "smote":
            return SmoteGenerator()
        case "ctgan":
            return CTGANGenerator()
        case "tabpfnunsupervised":
            return FullTabpfnGen()
        case "tabiclgen":
            return TabPFGenClassifier(
                n_sgld_steps=100, clasifier_class=TabICLClassifier
            )
        case "tabpfngen":
            return TabPFGenClassifier(n_sgld_steps=100)
        case "nf":
            return NeuralSplineFlowsGenerator()
        case _:
            raise Exception("Chosen generator type is incorrect")


def get_regression_model(args):
    match args.generator_type.lower():
        case "smote":
            return SmoterGenerator()
        case "ctgan":
            return CTGANGenerator(preprocess=True)
        case "tabpfnunsupervised":
            return FullTabpfnGen()
        case "tabiclgen":
            return TabPFGenRegressor(n_sgld_steps=100, clasifier_class=TabICLClassifier)
        case "tabpfngen":
            return TabPFGenRegressor(n_sgld_steps=100)
        case "nf":
            return NeuralSplineFlowsGenerator()
        case _:
            raise Exception("Chosen generator type is incorrect")


def get_metrics_to_compute(args):
    metrics = {}
    if args.k_anonimity:
        metrics["k-anonimity"] = metric_wrappers.KAnonimity()
    if args.unlinkability:
        metrics["unlinkability"] = metric_wrappers.Unlinkability()
    if args.distance_to_nearest:
        metrics["distance-to-nearest"] = metric_wrappers.DistanceToNearestNeighbour()
    if args.dataset_statistics:
        metrics["dataset-statistics"] = metric_wrappers.DatasetStatistics()
    if args.tree_depth_precision_relation:
        metrics["tree-depth-precision-relation"] = metric_wrappers.MinimalTree()
    if args.area_under_curve:
        metrics["area-under-curve"] = metric_wrappers.ModelAuc()
    if args.convex_hull:
        metrics["convex-hull"] = metric_wrappers.ConvexHull()
    if args.svn_discrimination:
        metrics["svn-discrimination"] = metric_wrappers.Discrimination()
    return metrics


def generate_samples(
    real_dataset: pd.DataFrame,
    target: str,
    model,
    **kwargs,
):
    real_x, real_y = (
        real_dataset.drop(target, axis=1),
        real_dataset[target].to_list(),
    )

    synth_x, synth_y = model(
        real_x,
        real_y,
        n_samples=real_x.shape[0],
        balance_classes=True,
        **kwargs,
    )
    synth_x = pd.DataFrame(synth_x, columns=real_x.columns)
    synth_x[target] = synth_y
    return synth_x


def generate_model_metrics(
    model,
    datasets: dict[str, Callable],
    current_output_path: Path,
    target: str,
    metrics: list[metric_wrappers.MetricWrapper],
    number_of_repetitions: int,
):
    for dataset_name, dataset_getter in datasets.items():
        (current_output_path / dataset_name).mkdir(exist_ok=True, parents=True)
        for run_number in range(number_of_repetitions):
            current_run_results = {}
            train, test = dataset_getter()
            synth = generate_samples(train, target, model)
            for metric_name in metrics:
                current_run_results[metric_name] = metrics[metric_name](
                    synthetic=synth,
                    real_train=train,
                    real_test=test,
                    target=target,
                )
            with open(
                (current_output_path / f"{dataset_name}/{run_number}.json"), "w"
            ) as json_file:
                json.dump(current_run_results, json_file, indent=4)


def main():
    args = parse_args()
    model = get_clasification_model(args)
    current_output_path: Path = args.output_dir / datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    current_output_path.mkdir(exist_ok=True, parents=True)
    save_args(args, current_output_path)
    metrics: list[metric_wrappers.MetricWrapper] = get_metrics_to_compute(args)
    generate_model_metrics(
        model,
        AVAILABLE_CLASSIFICATION_DATASETS,
        current_output_path,
        clasification_datasets.CLASYFICATION_TARGET,
        metrics,
        args.repetitions_number,
    )

    model = get_regression_model(args)

    generate_model_metrics(
        model,
        AVAILABLE_CLASSIFICATION_DATASETS,
        current_output_path,
        clasification_datasets.CLASYFICATION_TARGET,
        metrics,
        args.repetitions_number,
    )
