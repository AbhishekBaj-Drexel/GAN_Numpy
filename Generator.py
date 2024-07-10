from ActivationFunctions import ActivationFunctions
from BatchNormalization import BatchNormalization
import numpy as np

class Generator:
    # Initialize the Generator with latent dimension and activation functions.
    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim
        self.activation_fn = ActivationFunctions()
        self.batch_norm = BatchNormalization()
        self.g = self.init_generator_weights()

    # Initialize weights for the Generator.
    def init_generator_weights(self):
        return {
            'Whl1': np.random.randn(256, 100) * 0.01,
            'bhl1': np.zeros((256, 1)),
            'Gbn1': np.ones((256, 1)),
            'Bbn1': np.zeros((256, 1)),
            'Whl2': np.random.randn(512, 256) * 0.01,
            'bhl2': np.zeros((512, 1)),
            'Gbn2': np.ones((512, 1)),
            'Bbn2': np.zeros((512, 1)),
            'Whl3': np.random.randn(1024, 512) * 0.01,
            'bhl3': np.zeros((1024, 1)),
            'Gbn3': np.ones((1024, 1)),
            'Bbn3': np.zeros((1024, 1)),
            'Wout': np.random.randn(784, 1024) * 0.01,
            'bout': np.zeros((784, 1)),
            'Gout': np.ones((784, 1)),
            'Bout': np.zeros((784, 1))
        }

    def get_output(self, n_outputs=1, train=False):
        act = {}
        act['noise_vector'] = np.random.normal(0, 1, (self.latent_dim, n_outputs)) * 1e-4
        act['z_hl1'] = np.dot(self.g['Whl1'], act['noise_vector']) + self.g['bhl1']
        act['a_hl1'] = self.activation_fn.leaky_relu(act['z_hl1'])
        act['bn_hl1'] = self.batch_norm.batch_norm_forward(act['a_hl1'], self.g['Gbn1'], self.g['Bbn1'], 'bn1', train)

        act['z_hl2'] = np.dot(self.g['Whl2'], act['bn_hl1']['output']) + self.g['bhl2']
        act['a_hl2'] = self.activation_fn.leaky_relu(act['z_hl2'])
        act['bn_hl2'] = self.batch_norm.batch_norm_forward(act['a_hl2'], self.g['Gbn2'], self.g['Bbn2'], 'bn2', train)

        act['z_hl3'] = np.dot(self.g['Whl3'], act['bn_hl2']['output']) + self.g['bhl3']
        act['a_hl3'] = self.activation_fn.leaky_relu(act['z_hl3'])
        act['bn_hl3'] = self.batch_norm.batch_norm_forward(act['a_hl3'], self.g['Gbn3'], self.g['Bbn3'], 'bn3', train)

        act['z_out'] = np.dot(self.g['Wout'], act['bn_hl3']['output']) + self.g['bout']
        act['bn_out'] = self.batch_norm.batch_norm_forward(act['z_out'], self.g['Gout'], self.g['Bout'], 'bnout', train)

        act['a_out'] = self.activation_fn.tanh(act['bn_out']['output'])
        return act

    # Clip gradients to prevent exploding gradients.
    def clip_gradients(self, gradients, clip_value=1.0):
        for key in gradients:
            np.clip(gradients[key], -clip_value, clip_value, out=gradients[key])
        return gradients
