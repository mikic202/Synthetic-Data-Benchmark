from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
import glob
from collections import defaultdict
import json


class RawData:
    def __init__(self, data_path: Path) -> None:
        self._data_path = data_path
        self._clasification_results = defaultdict(lambda: defaultdict(list))
        self._regression_results = defaultdict(lambda: defaultdict(list))
        self.load_data(self._data_path)

    def load_data(self, data_path: Path) -> None:
        for file in glob.glob(str(data_path / "clasification/*/*.json")):
            dataset_name = file.split("/")[-2]
            with open(file) as json_file:
                json_data = json.load(json_file)
                self.append_clasification_results(json_data, dataset_name)

        for file in glob.glob(str(data_path / "regression/*/*.json")):
            dataset_name = file.split("/")[-2]
            with open(file) as json_file:
                json_data = json.load(json_file)
                self.append_regression_results(json_data, dataset_name)

    @property
    def results(self) -> dict[str, dict]:
        return {**self._clasification_results, **self._regression_results}

    @property
    def clasification_results(self) -> dict[str, dict]:
        return self._clasification_results

    @property
    def regression_results(self) -> dict[str, dict]:
        return self._regression_results

    def append_clasification_results(
        self, file_results: dict[str, Any], dataset_name: str
    ):
        for metric in file_results:
            self._clasification_results[metric][dataset_name].append(
                file_results[metric]
            )

    def append_regression_results(
        self, file_results: dict[str, Any], dataset_name: str
    ):
        for metric in file_results:
            self._regression_results[metric][dataset_name].append(file_results[metric])


class BasePostprocessor(ABC):
    @abstractmethod
    def __call__(self, raw_data: RawData) -> Any:
        pass
