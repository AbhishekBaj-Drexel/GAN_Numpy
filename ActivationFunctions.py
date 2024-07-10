import numpy as np



class ActivationFunctions:
    @staticmethod
    def leaky_relu(x, alpha=0.2):
        return np.where(x > 0, x, x * alpha)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)