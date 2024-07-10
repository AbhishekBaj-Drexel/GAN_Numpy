from ActivationFunctions import ActivationFunctions
from BatchNormalization import BatchNormalization
import numpy as np

class Discriminator:
    # Initialize the Discriminator with activation functions and weights.
    def __init__(self):
        self.activation_fn = ActivationFunctions()
        self.d = self.init_discriminator_weights()
    # Initialize weights for the Discriminator.
    def init_discriminator_weights(self):
        return {
            'Whl1': np.random.randn(512, 784) * 0.01,
            'bhl1': np.zeros((512, 1)),
            'Whl2': np.random.randn(256, 512) * 0.01,
            'bhl2': np.zeros((256, 1)),
            'Wout': np.random.randn(1, 256) * 0.01,
            'bout': np.zeros((1, 1))
        }
    # Perform forward pass of the Discriminator.
    def get_output(self, input_batch):
        act = {}
        act['input'] = input_batch
        act['z_hl1'] = np.dot(self.d['Whl1'], act['input']) + self.d['bhl1']
        act['a_hl1'] = self.activation_fn.leaky_relu(act['z_hl1'])

        act['z_hl2'] = np.dot(self.d['Whl2'], act['a_hl1']) + self.d['bhl2']
        act['a_hl2'] = self.activation_fn.leaky_relu(act['z_hl2'])

        act['z_out'] = np.dot(self.d['Wout'], act['a_hl2']) + self.d['bout']
        act['a_out'] = self.activation_fn.sigmoid(act['z_out'])

        return act
    # Clip gradients to prevent exploding gradients.
    def clip_gradients(self, gradients, clip_value=1.0):
        for key in gradients:
            np.clip(gradients[key], -clip_value, clip_value, out=gradients[key])
        return gradients