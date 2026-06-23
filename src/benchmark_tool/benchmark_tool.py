from src.benchmark_tool.argparser import parse_args, save_args
from src.test_datasets import clasification_datasets, regression_datasets
from src.benchmark_tool import metric_wrappers
import pandas as pd
import datetime
from pathlib import Path
import json
from typing import Callable
from src.constants import (
    K_ANONIMITY,
    CONVEX_HULL,
    DISTANCE_TO_NEAREST_NEIGHBOUR,
    DISCRIMINATION,
    TREE_DEPTH_RELATION,
    UNLINKABILITY,
    K_ANONIMITY_WITH_REAL
)
import tqdm

AVAILABLE_CLASSIFICATION_DATASETS = {
    "mfeat_zernike": clasification_datasets.get_mfeat_zernike_dataset,
    "pc4": clasification_datasets.get_pc4_dataset,
    "climate_model_simulation": clasification_datasets.get_climate_model_simulation_dataset,
    "wdbc": clasification_datasets.get_wdbc_dataset,
    "analcatdata_authorship": clasification_datasets.get_analcatdata_authorship_dataset,
    "heart_diseasee_dataset": clasification_datasets.get_heart_disease_dataset,
    # "pulsar": clasification_datasets.get_pulsar_dataset,
    # "cardiovascular": clasification_datasets.get_cardiovascular_dataset,
}


AVALIABLE_REGRESSION_DATASETS = {
    "heart_failure_clinical_regresion": regression_datasets.get_heart_failure_clinical_regresion_dataset,
    "sleep_deprivation_and_cognitive_performance_regression": regression_datasets.get_sleep_deprivation_and_cognitive_performance_regression_dataset,
    "house_prices_regression": regression_datasets.get_house_prices_regression_dataset,
    "abalone": regression_datasets.get_abalone_dataset,
    # "superconduct_regression": regression_datasets.get_superconduct_regression_dataset,
}


def get_clasification_model(args):
    match args.generator_type.lower():
        case "smote":
            from src.model_wrappers.smote_generator import SmoteGenerator
            return SmoteGenerator()
        case "ctgan":
            from src.model_wrappers.ctgan_generator import CTGANGenerator
            return CTGANGenerator()
        case "tabpfnunsupervised":
            from src.model_wrappers.full_tabpfn_gen import FullTabpfnGen
            return FullTabpfnGen()
        case "tabiclgen":
            from external.tab_pfn_gen.src.tabpfgen.tabpfgen import (
                TabPFGenClassifier,
                TabICLClassifier,
                TabICLRegressor)
            return TabPFGenClassifier(
                n_sgld_steps=100,
                clasifier_class=TabICLClassifier,
                regressor_class=TabICLRegressor,
            )
        case "tabpfngen":
            from external.tab_pfn_gen.src.tabpfgen.tabpfgen import (
                TabPFGenClassifier)
            return TabPFGenClassifier(n_sgld_steps=100)
        case "nf":
            from src.model_wrappers.neural_spline_flows_generator import NeuralSplineFlowsGenerator
            return NeuralSplineFlowsGenerator()
        case "real":
            from src.model_wrappers.real_data_generator import RealDataGenerator
            return RealDataGenerator()
        case "random":
            from src.model_wrappers.random_generator import RandomGenerator
            return RandomGenerator()
        case _:
            raise Exception("Chosen generator type is incorrect")


def get_regression_model(args):
    match args.generator_type.lower():
        case "smote":
            from src.model_wrappers.smote_generator import SmoterGenerator
            return SmoterGenerator()
        case "ctgan":
            from src.model_wrappers.ctgan_generator import CTGANGenerator
            return CTGANGenerator(preprocess=True)
        case "tabpfnunsupervised":
            from src.model_wrappers.full_tabpfn_gen import FullTabpfnGen
            return FullTabpfnGen()
        case "tabiclgen":
            from external.tab_pfn_gen.src.tabpfgen.tabpfgen import (
                TabPFGenRegressor,
                TabICLClassifier,
                TabICLRegressor)
            return TabPFGenRegressor(n_sgld_steps=100, clasifier_class=TabICLClassifier, regressor_class=TabICLRegressor)
        case "tabpfngen":
            from external.tab_pfn_gen.src.tabpfgen.tabpfgen import (
                TabPFGenRegressor)
            return TabPFGenRegressor(n_sgld_steps=100)
        case "nf":
            from src.model_wrappers.neural_spline_flows_generator import NeuralSplineFlowsGenerator
            return NeuralSplineFlowsGenerator()
        case "real":
            from src.model_wrappers.real_data_generator import RealDataGenerator
            return RealDataGenerator()
        case "random":
            from src.model_wrappers.random_generator import RandomGenerator
            return RandomGenerator()
        case _:
            raise Exception("Chosen generator type is incorrect")


def get_metrics_to_compute(args):
    metrics = {}
    if args.k_anonimity:
        metrics[K_ANONIMITY] = metric_wrappers.KAnonimity()
        metrics[K_ANONIMITY_WITH_REAL] = metric_wrappers.KAnonimityWithReal()
    if args.unlinkability:
        metrics[UNLINKABILITY] = metric_wrappers.Unlinkability()
    if args.distance_to_nearest:
        metrics[DISTANCE_TO_NEAREST_NEIGHBOUR] = (
            metric_wrappers.DistanceToNearestNeighbour()
        )
        metrics["distance_to_nearest_real_neighbour"] = (
            metric_wrappers.DistanceToNearestRealNeighbour()
        )
    if args.dataset_statistics:
        metrics["dataset-statistics"] = metric_wrappers.DatasetStatistics()
    if args.tree_depth_precision_relation:
        metrics[TREE_DEPTH_RELATION] = metric_wrappers.MinimalTree()
    if args.area_under_curve:
        metrics["area-under-curve"] = metric_wrappers.ModelAuc()
    if args.convex_hull:
        metrics[CONVEX_HULL] = metric_wrappers.ConvexHull()
    if args.svn_discrimination:
        metrics[DISCRIMINATION] = metric_wrappers.Discrimination()
    if args.identity:
        metrics["identity"] = metric_wrappers.Identity()
    return metrics


def generate_samples(
    real_dataset: pd.DataFrame,
    target: str,
    model,
    **kwargs,
):
    real_x, real_y = (
        real_dataset.drop(target, axis=1),
        real_dataset[target].to_numpy(),
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
    progress_bar = tqdm.tqdm(datasets.items())
    for dataset_name, dataset_getter in progress_bar:
        (current_output_path / dataset_name).mkdir(exist_ok=True, parents=True)
        for run_number in range(number_of_repetitions):
            progress_bar.set_description(f"Testing dataset {dataset_name}, iteration {run_number}/{number_of_repetitions}")
            current_run_results = {}
            train, test = dataset_getter()
            synth = generate_samples(train, target, model)
            synth.to_csv(current_output_path / f"{dataset_name}/{run_number}.csv")
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
    main_output_path: Path = args.output_dir / datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    main_output_path.mkdir(exist_ok=True, parents=True)
    save_args(args, main_output_path)
    current_output_path: Path = main_output_path / "clasification"
    current_output_path.mkdir(exist_ok=True, parents=True)
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
    current_output_path: Path = main_output_path / "regression"
    current_output_path.mkdir(exist_ok=True, parents=True)
    generate_model_metrics(
        model,
        AVALIABLE_REGRESSION_DATASETS,
        current_output_path,
        regression_datasets.REGRESION_TARGET,
        metrics,
        args.repetitions_number,
    )
