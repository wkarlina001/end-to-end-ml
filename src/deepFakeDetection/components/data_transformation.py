from deepFakeDetection.entity.config_entity import DataTransformationConfig
import os
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
from ensure import ensure_annotations
from deepFakeDetection import logger
import numpy as np

class DataTransformation:
    def __init__(self, config:DataTransformationConfig):
        self.config = config
        
    def create_spectrogram(self, audio_file, image_file):
        '''
        Create spectogram from input audio file
        '''
        try:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

            y, sr = librosa.load(audio_file)
            ms = librosa.feature.melspectrogram(y=y, sr=sr)
            log_ms = librosa.power_to_db(ms, ref=np.max)
            librosa.display.specshow(log_ms, sr=sr)

            fig.savefig(image_file)
            plt.close(fig)
        except Exception as e:
            raise e
    
    def create_pngs_from_wavs(self, src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)

        dir = os.listdir(src)
        logger.info(f"Transform data from {src} into {dest}")
        for i, file in enumerate(dir):
            input_file = os.path.join(src, file)
            output_file = os.path.join(dest, file.replace('.wav', '.png'))
            try:
                if not os.path.exists(output_file):
                    self.create_spectrogram(input_file, output_file)
            except Exception as e:
                continue
                # raise e
        logger.info(f"Transformed data from {src} into {dest}")

    def transform_all_data(self):
        try:
            sub_dir_list = ["fake", "real"]
            dirs = os.listdir(self.config.input_dir)
            for dir in dirs:
                for sub_dir in sub_dir_list:
                    src = os.path.join(self.config.input_dir, dir, sub_dir)
                    dest = os.path.join(self.config.output_dir, dir, sub_dir)
                    self.create_pngs_from_wavs(src, dest)

        except Exception as e:
            raise e                
