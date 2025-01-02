from deepFakeDetection.config.configuration import ConfigurationManager
from deepFakeDetection.components.data_transformation import DataTransformation
from deepFakeDetection import logger

STAGE_NAME = "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()

        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.transform_all_data()

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
    