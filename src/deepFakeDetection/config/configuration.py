from deepFakeDetection.constants import *
from deepFakeDetection.utils.common import read_yaml, create_directories
from deepFakeDetection.entity.config_entity import *

class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        create_directories([self.config.artifact_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        data_transformation_config = DataTransformationConfig(
            input_dir=config.input_dir,
            output_dir=config.output_dir
        )

        return data_transformation_config
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        
        training_config = ModelTrainingConfig(
            spectogram_data_dir = self.config.spectogram_data_dir,
            train_model_path = self.config.train_model_path,
            accuracy_plot_path = self.config.accuracy_plot_path,
            params_epochs = self.params.EPOCHS,
            params_batch_size = self.params.BATCH_SIZE,
            params_image_size = self.params.IMAGE_SIZE,
            params_learning_rate = self.params.LEARNING_RATE
        )

        return training_config