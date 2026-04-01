from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
import glob
from collections import defaultdict
import json


class RawData:
    def __init__(self, data_path: Path) -> None:
        self._data_path = data_path
        self.load_data(self._data_path)

    def load_data(self, data_path: Path) -> None:
        self._clasification_results = defaultdict(list)
        for file in glob.glob(str(data_path / "clasification/*/*.json")):
            dataset_name = file.split("/")[-2]
            with open(file) as json_file:
                json_data = json.load(json_file)
                self._clasification_results[dataset_name].append(json_data)

        self._regression_results = defaultdict(list)
        for file in glob.glob(str(data_path / "regression/*/*.json")):
            dataset_name = file.split("/")[-2]
            with open(file) as json_file:
                json_data = json.load(json_file)
                self._regression_results[dataset_name].append(json_data)

    @property
    def results(self) -> dict[str, dict]:
        return {**self._clasification_results, **self._regression_results}

    @property
    def clasification_results(self) -> dict[str, dict]:
        return self._clasification_results

    @property
    def regression_results(self) -> dict[str, dict]:
        return self._regression_results


class BasePostprocessor(ABC):
    @abstractmethod
    def __call__(self, raw_data: RawData) -> Any:
        pass
