import numpy as np

class BatchNormalization:
    # Initialize batch normalization parameters.
    def __init__(self, BN_ALPHA=0.9, bn_epsilon=1e-5):
        self.BN_ALPHA = BN_ALPHA
        self.bn_epsilon = bn_epsilon
        self.bnStored = self.init_batch_norm_stores()

    # Initialize stores for batch normalization means and variances.
    def init_batch_norm_stores(self):
        bnStored = {
            'bn1': {'means': 0, 'variances': 0},
            'bn2': {'means': 0, 'variances': 0},
            'bn3': {'means': 0, 'variances': 0},
            'bnout': {'means': 0, 'variances': 0}
        }
        return bnStored

    # Perform forward pass of batch normalization.
    def batch_norm_forward(self, input, gamma, beta, layer_name, train=True):
        bn = {}
        if train:
            bn['means'] = np.mean(input, axis=1, keepdims=True)
            bn['variances'] = np.var(input, axis=1, keepdims=True)
            self.bnStored[layer_name]['means'] = self.bnStored[layer_name]['means'] * self.BN_ALPHA + (1 - self.BN_ALPHA) * bn['means']
            self.bnStored[layer_name]['variances'] = self.bnStored[layer_name]['variances'] * self.BN_ALPHA + (1 - self.BN_ALPHA) * bn['variances']
        else:
            bn['means'] = self.bnStored[layer_name]['means']
            bn['variances'] = self.bnStored[layer_name]['variances']

        bn['input_normalized'] = (input - bn['means']) / np.sqrt(bn['variances'] + self.bn_epsilon)
        bn['output'] = gamma * bn['input_normalized'] + beta
        return bn
