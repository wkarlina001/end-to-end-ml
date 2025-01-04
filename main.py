from deepFakeDetection import logger
from deepFakeDetection.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from deepFakeDetection.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from deepFakeDetection.pipeline.stage_03_model_training import ModelTrainingPipeline
from deepFakeDetection.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline

if __name__ == '__main__':
    # STAGE_NAME = "Data Ingestion Stage"
    # try:    
    #     logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    #     obj = DataIngestionTrainingPipeline()
    #     obj.main()
    #     logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    # except Exception as e:
    #     logger.exception(e)
    #     raise e
    
    # STAGE_NAME = "Data Transformation Stage"
    # try:
    #     logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    #     obj = DataTransformationTrainingPipeline()
    #     obj.main()
    #     logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    # except Exception as e:
    #     logger.exception(e)
    #     raise e
    
    # STAGE_NAME = "Model Training Stage"
    # try:
    #     logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    #     obj = ModelTrainingPipeline()
    #     obj.main()
    #     logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    # except Exception as e:
    #     logger.exception(e)
    #     raise e
    
    STAGE_NAME = "Model Evaluation Stage"
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    except Exception as e:
        logger.exception(e)
        raise e