from deepFakeDetection.constants import *
from deepFakeDetection.utils.common import read_yaml, create_directories
from deepFakeDetection.entity.config_entity import *
import numpy as np
import librosa.display, os
import matplotlib.pyplot as plt
from pathlib import Path
from ensure import ensure_annotations
from deepFakeDetection import logger
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from tensorflow.keras.utils import load_img, img_to_array 
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

class Training:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def get_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.config.params_image_size))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(2, activation='sigmoid'))
        optimizer = keras.optimizers.Adam(lr=self.config.params_learning_rate)
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

        print(self.model.summary())
    
    def load_images_from_path(self, path, label):
        images = []
        labels = []

        for file in os.listdir(path):
            images.append(img_to_array(load_img(os.path.join(path, file), target_size=(self.config.params_image_size))))
            labels.append((label))
            
        return images, labels

    def create_dataset(self, subdir):
        images, labels = self.load_images_from_path(os.path.join(self.config.spectogram_data_dir, subdir, "real"), 1)
        x += images
        y += labels

        images, labels = self.load_images_from_path(os.path.join(self.config.spectogram_data_dir, subdir, "fake"), 0)
        x += images
        y += labels
        
        return x, y
    
    def train(self):
        x_train, y_train = self.create_dataset("training")
        x_dev, y_dev = self.create_dataset("validation")

        x_train_norm = np.array(x_train) / 255 # normalise
        x_dev_norm = np.array(x_dev) / 255

        y_train_encoded = to_categorical(y_train)
        y_dev_encoded = to_categorical(y_dev)

        self.hist = self.model.fit(x_train_norm, y_train_encoded,
                              validation_data=(x_dev_norm, y_dev_encoded),
                              batch_size=self.config.params_batch_size, epochs=self.config.params_epochs)
        self.save_model(
            path=self.config.train_model_path,
            model=self.model
        )
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def plot_accuracy(self):
        acc = self.hist.history['accuracy']
        val_acc = self.hist.history['val_accuracy']
        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, '-', label='Training Accuracy')
        plt.plot(epochs, val_acc, ':', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()
        plt.savefig(self.config.accuracy_plot_path)
        plt.close()
