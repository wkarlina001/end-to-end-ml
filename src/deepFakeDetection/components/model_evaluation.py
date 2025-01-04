import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from tensorflow.keras.utils import to_categorical
from deepFakeDetection.constants import *
from deepFakeDetection.utils.common import read_yaml, create_directories, save_json
from deepFakeDetection.entity.config_entity import *
import os

os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/wkarlina001/end-to-end-ml.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="wkarlina001"
os.environ["MLFLOW_TRACKING_PASSWORD"]="9e92f54fc366d8f0f6d5e6a7adce34c1a1d51177"


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def load_images_from_path(self, path, label):
        images = []
        labels = []

        for file in os.listdir(path):
            try:
                images.append(img_to_array(load_img(os.path.join(path, file), target_size=(self.config.params_image_size))))
                labels.append((label))
            except:
                continue
        return images, labels

    def create_dataset(self, subdir):
        print(os.path.join(self.config.spectogram_data_dir, subdir, "real"))
        x = []
        y = []
        images, labels = self.load_images_from_path(os.path.join(self.config.spectogram_data_dir, subdir, "real"), 1)
        x += images
        y += labels

        images, labels = self.load_images_from_path(os.path.join(self.config.spectogram_data_dir, subdir, "fake"), 0)
        x += images
        y += labels
        
        return x, y

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        # self.model = tf.keras.models.load_model("../artifacts/training/model.h5")
        
        x, y = self.create_dataset("testing")
        x1 = np.array(x) / 255
        y1 = to_categorical(y)

        self.score = self.model.evaluate(x1, y1, batch_size=128)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="CustomCNNModel")
            else:
                mlflow.keras.log_model(self.model, "model")

