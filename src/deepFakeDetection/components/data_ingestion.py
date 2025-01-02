import os
import urllib.request as request
import zipfile
from deepFakeDetection import logger
from deepFakeDetection.utils.common import get_size
import tarfile
from deepFakeDetection.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config:DataIngestionConfig):
        self.config = config

    def unzip_data_file(self) -> str:
        '''
        Fetch data from the url
        '''
        try:
            local_dir_data = self.config.unzip_dir
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Unzipping data from {zip_download_dir} into file {zip_download_dir}")

            tar = tarfile.open(zip_download_dir, 'r:gz')
            tar.extractall(local_dir_data)
            tar.close()

            logger.info(f"Unzipped data from {zip_download_dir} into file {zip_download_dir}")

        except Exception as e:
            raise e

