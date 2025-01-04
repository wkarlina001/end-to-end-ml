from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    input_dir: Path
    output_dir: Path

@dataclass(frozen=True)
class ModelTrainingConfig:
    spectogram_data_dir: Path
    train_model_path: Path
    accuracy_plot_path: Path
    params_epochs: int
    params_batch_size: int
    params_image_size: list
    params_learning_rate: float

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    spectogram_data_dir: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int
