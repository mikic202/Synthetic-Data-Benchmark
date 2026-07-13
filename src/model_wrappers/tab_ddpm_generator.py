import pandas as pd
import numpy as np
import toml
from src.metrics import (
    NUMBER_OF_UNIQUE_ELEMENTS_FOR_CLASIFICATION,
)
from pathlib import Path
from sklearn.model_selection import train_test_split
import subprocess
import os
from external.tabddpm.lib.util import TaskType
import json
import shutil


class TabDDPMGenerator:
    def __call__(self,
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        n_samples: int,
        config_filepath: str = "src/model_wrappers/default_tab_ddpm_config.toml",
        **kwargs,):
        default_config = self.__load_default_config(config_filepath)
        default_config["sample"]["num_samples"] = n_samples
        unique_y_values = len(np.unique(y_train))
        if unique_y_values == 2:
            task_type = str(TaskType.BINCLASS)
        elif unique_y_values <= NUMBER_OF_UNIQUE_ELEMENTS_FOR_CLASIFICATION:
            task_type = str(TaskType.MULTICLASS)
        else:
            task_type = str(TaskType.REGRESSION)
        self.__save_dataset_info(Path(default_config["parent_dir"])/"info.json", n_samples, task_type)
        default_config["sample"]["batch_size"] = n_samples // 3 + 1
        num_of_categorical_features = sum([1 for unique_elements in x_train.nunique() if unique_elements <= NUMBER_OF_UNIQUE_ELEMENTS_FOR_CLASIFICATION])
        default_config["num_numerical_features"] = x_train.shape[1] - num_of_categorical_features
        if len(np.unique(y_train)) <= NUMBER_OF_UNIQUE_ELEMENTS_FOR_CLASIFICATION:
            default_config["model_params"]["is_y_cond"] = True
            default_config["model_params"]["num_classes"] = len(np.unique(y_train))
        default_config["model_params"]["d_in"] = sum([1 if unique_elements > NUMBER_OF_UNIQUE_ELEMENTS_FOR_CLASIFICATION else unique_elements for unique_elements in x_train.nunique()])
        self.__save_config(Path(default_config["parent_dir"])/"config.toml", default_config)
        self.__save_data_for_processing(Path(default_config["real_data_path"]), x_train, y_train)

        run_command = f"PYTHONPATH={os.getcwd()}/external/tabddpm uv run python {os.getcwd()}/external/tabddpm/scripts/pipeline.py --config {str(Path(default_config['parent_dir'])/'config.toml')} --sample --train"
        process = subprocess.Popen(run_command, shell=True, stdout=subprocess.PIPE)
        _, _ = process.communicate()
        process.wait()

        x_synth = self.__load_features(Path(default_config["parent_dir"]), x_train)
        y_synth = np.load(Path(default_config["parent_dir"]) / "y_train.npy")
        shutil.rmtree(Path(default_config["parent_dir"]))
        return x_synth, y_synth

    def __load_features(self, parent_dir: Path, x_train: pd.DataFrame):
        new_column_order = [i for i, unique_elements in enumerate(x_train.nunique()) if unique_elements <= NUMBER_OF_UNIQUE_ELEMENTS_FOR_CLASIFICATION] + [i for i, unique_elements in enumerate(x_train.nunique()) if unique_elements > NUMBER_OF_UNIQUE_ELEMENTS_FOR_CLASIFICATION]
        new_column_order = x_train.columns[new_column_order]
        list_of_features = []
        if (parent_dir / "X_cat_train.npy").exists():
            list_of_features.append(np.load(parent_dir / "X_cat_train.npy", allow_pickle=True))
        if (parent_dir / "X_num_train.npy").exists():
            list_of_features.append(np.load(parent_dir / "X_num_train.npy", allow_pickle=True))
        return pd.DataFrame(np.concatenate(list_of_features, axis=1), columns=new_column_order).astype(float)[x_train.columns]

    def __save_data_for_processing(self, path, x_train: pd.DataFrame,
        y_train: np.ndarray,):
        categorical_feature_mask  = [i for i, unique_elements in enumerate(x_train.nunique()) if unique_elements <= NUMBER_OF_UNIQUE_ELEMENTS_FOR_CLASIFICATION]
        numerical_feature_mask  = [i for i, unique_elements in enumerate(x_train.nunique()) if unique_elements > NUMBER_OF_UNIQUE_ELEMENTS_FOR_CLASIFICATION]

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
        np.save(path / "y_train.npy", y_train)
        np.save(path / "y_test.npy", y_train)
        np.save(path / "y_val.npy", y_val)
        self.__save_feature_vector(path, x_train, "train", categorical_feature_mask, numerical_feature_mask)
        self.__save_feature_vector(path, x_train, "test", categorical_feature_mask, numerical_feature_mask)
        self.__save_feature_vector(path, x_val, "val", categorical_feature_mask, numerical_feature_mask)


    def __save_feature_vector(self, path: Path, x_train: pd.DataFrame, split_name: str, categorical_feature_mask, numerical_feature_mask):
        if len(categorical_feature_mask) > 0:
            np.save(path / f"X_cat_{split_name}.npy", x_train.values[:, categorical_feature_mask])
        if len(numerical_feature_mask) > 0:
            np.save(path / f"X_num_{split_name}.npy", x_train.values[:, numerical_feature_mask])


    def __save_config(self, path: Path, config):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            toml.dump(config, f)

    def __save_dataset_info(self, path, datset_size, task_type):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"task_type": task_type, "n_classes": datset_size}
        with open(path, "w") as f:
            json.dump(data, f)

    def __load_default_config(self, config_filepath: str = "src/model_wrappers/default_tab_ddpm_config.toml"):
        with open(config_filepath, "r") as f:
            data = toml.load(f)
        return data
