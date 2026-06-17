from pathlib import Path
import glob
import numpy as np
import statistics
import pandas as pd
from src.benchmark_tool.benchmark_tool import (
    AVAILABLE_CLASSIFICATION_DATASETS,
    AVALIABLE_REGRESSION_DATASETS,
)
import re
from collections import defaultdict
from scipy.spatial.distance import jensenshannon

dataset_names = list(AVAILABLE_CLASSIFICATION_DATASETS.keys()) + list(
    AVALIABLE_REGRESSION_DATASETS.keys()
)


class DataDistributionAggregator:
    def __init__(
        self,
        input_paths: list[Path],
        reference_data_path: Path,
        output_path: Path,
        generator_types: list[str],
    ) -> None:
        self._generator_types = generator_types
        self._output_path = output_path
        self._filename = "dataset-distribution"
        self._csv_file_paths = {}
        for generator_type, data_path in zip(
            self._generator_types,
            input_paths,
        ):
            self._csv_file_paths[generator_type] = self._groupby_dataset(
                list(glob.glob(str(data_path / f"**/*.csv"), recursive=True))
            )
        self._reference_csv_files = self._groupby_dataset(
            list(glob.glob(str(reference_data_path / f"**/*.csv"), recursive=True))
        )

    def _groupby_dataset(self, file_paths: list[str]):
        paths_split_into_datasets = defaultdict(list)
        regex = re.compile("|".join(map(re.escape, dataset_names)))
        for file in file_paths:
            dataset_match = regex.search(file)
            if dataset_match:
                paths_split_into_datasets[dataset_match.group()].append(file)
        return paths_split_into_datasets

    def _calculate_feature_js_distance(
        self, ref_col: pd.Series, synth_col: pd.Series
    ) -> float:
        ref_clean = ref_col.dropna()
        synth_clean = synth_col.dropna()

        if ref_clean.empty or synth_clean.empty:
            return 1.0

        if ref_clean.dtype == "object" or len(ref_clean.unique()) <= 15:
            all_categories = list(set(ref_clean.unique()) | set(synth_clean.unique()))
            ref_p = (
                ref_clean.value_counts().reindex(all_categories, fill_value=0).values
            )
            synth_p = (
                synth_clean.value_counts().reindex(all_categories, fill_value=0).values
            )
        else:
            combined = pd.concat([ref_clean, synth_clean])
            bins = np.linspace(combined.min(), combined.max(), num=50)

            ref_p, _ = np.histogram(ref_clean, bins=bins)
            synth_p, _ = np.histogram(synth_clean, bins=bins)

        ref_sum, synth_sum = ref_p.sum(), synth_p.sum()
        if ref_sum == 0 or synth_sum == 0:
            return 1.0

        ref_p = ref_p / ref_sum
        synth_p = synth_p / synth_sum

        js_dist = jensenshannon(ref_p, synth_p)
        if np.isnan(js_dist) or np.isinf(js_dist):
            return 1.0
        return js_dist

    def __call__(self) -> None:
        average_distance = defaultdict(lambda: defaultdict(list))
        for dataset in self._reference_csv_files:
            for algorithm in self._csv_file_paths:
                distribution_distances = []
                for ref_path, synth_path in zip(
                    self._reference_csv_files[dataset],
                    self._csv_file_paths[algorithm][dataset],
                ):
                    ref_values = pd.read_csv(ref_path)
                    synth_values = pd.read_csv(synth_path)
                    distribution_distances.append(
                        sum(
                            [
                                self._calculate_feature_js_distance(
                                    ref_values[collumn_name], synth_values[collumn_name]
                                )
                                for collumn_name in ref_values.columns
                            ]
                        )
                    )
                if distribution_distances:
                    average_distance[dataset][algorithm] = np.mean(
                        distribution_distances
                    )
        pd.DataFrame(average_distance).T.to_latex(
            self._output_path / f"{self._filename}.tex",
        )
