from src.benchmark_tool.argparser import parse_args
from src.model_wrappers.full_tabpfn_gen import FullTabpfnGen
from src.model_wrappers.smote_generator import SmoteGenerator
from src.model_wrappers.ctgan_generator import CTGANGenerator
from src.test_datasets import clasification_datasets, regression_datasets
from src.metrics.privacy_metrics import k_anonimity
import pandas as pd
import datetime
from pathlib import Path
import json


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


def get_model_class(args):
    match args.generator_type.lower():
        case "tabiclgen":
            pass
        case "smote":
            return SmoteGenerator
        case "ctgan":
            return CTGANGenerator
        case "tabpfnunsupervised":
            return FullTabpfnGen
        case _:
            raise Exception("Chosen generator type is incorrect")


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


def main():
    args = parse_args()
    generator_model_class = get_model_class(args)
    model = generator_model_class()
    current_output_path: Path = args.output_dir / datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    for dataset_name, dataset_getter in AVAILABLE_CLASSIFICATION_DATASETS.items():
        (current_output_path / dataset_name).mkdir(exist_ok=True, parents=True)
        for run_number in range(args.repetitions_number):
            current_run_results = {}
            train, test = dataset_getter()
            synth = generate_samples(
                train, clasification_datasets.CLASYFICATION_TARGET, model
            )
            current_run_results["k_anonimity"] = int(
                k_anonimity.calculate_k_anonimity_for_datset(synth)
            )
            with open(
                (current_output_path / f"{dataset_name}/{run_number}.json"), "w"
            ) as json_file:
                json.dump(current_run_results, json_file)
