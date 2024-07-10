from tensorflow.keras.datasets import mnist
import numpy as np


class DataLoader:
    def __init__(self, digit=0):
        (self.x_train, self.y_train), (_, _) = mnist.load_data()
        self.x_train = self.x_train[self.y_train == digit]
        self.x_train = self.x_train / 127.5 - 1.  # Normalize to [-1, 1]
        self.x_train = np.expand_dims(self.x_train, axis=3)

    def get_data(self):
        return self.x_train

