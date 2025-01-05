import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from tensorflow.keras.utils import load_img, img_to_array
import librosa
import matplotlib.pyplot as plt

class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


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
        
        return image_file
    
    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts","training","model.h5"))

        # image_file = self.create_spectrogram(self.filename, self.filename.replace(".wav",".png"))
        test_image = image.load_img(self.filename, target_size = (224,224,3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 1:
            prediction = 'Fake'
            return [{ "image" : prediction}]
        else:
            prediction = 'Real'
            return [{ "image" : prediction}]