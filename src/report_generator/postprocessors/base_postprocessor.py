from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
import os
from collections import defaultdict
import json


class RawData:
    def __init__(self, data_path: Path) -> None:
        self._data_path = data_path

    def load_data(self, data_path: Path) -> None:
        self._clasification_results = defaultdict(list)
        for file in os.walk(data_path / "clasification/*.json"):
            dataset_name = file.split("/")[-2]
            json_data = json.load(file)
            self._clasification_results[dataset_name].append(json_data)

        self._regression_results = defaultdict(list)
        for file in os.walk(data_path / "regression/*.json"):
            dataset_name = file.split("/")[-2]
            json_data = json.load(file)
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
