from deepFakeDetection.config.configuration import ConfigurationManager
from deepFakeDetection.components.model_evaluation import Evaluation
from deepFakeDetection import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()

        evaluation = Evaluation(config=model_evaluation_config)
        
        evaluation.evaluation()
        evaluation.log_into_mlflow()

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
    
