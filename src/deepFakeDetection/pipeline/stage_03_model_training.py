from deepFakeDetection.config.configuration import ConfigurationManager
from deepFakeDetection.components.model_training import Training
from deepFakeDetection import logger

STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()

        training = Training(config=model_training_config)
        training.get_model()
        training.train()
        training.plot_accuracy()

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
    
