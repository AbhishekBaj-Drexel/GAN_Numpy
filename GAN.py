import numpy as np
from DataLoader import DataLoader
import matplotlib.pyplot as plt
import os
from ActivationFunctions import ActivationFunctions
from BatchNormalization import BatchNormalization
from Generator import Generator
from Discriminator import Discriminator

class GAN:
    # Initialize GAN with data loader and model parameters.
    def __init__(self, data_loader):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        self.x_train = data_loader.get_data()
        print(f"Data Loaded: {self.x_train.shape}")
        self.all_data = np.reshape(self.x_train, (-1, 784)).T

        self.batch_size = 64
        self.lr = 0.0001  # Reduced learning rate
        self.small_value = 1e-25
        self.bn_epsilon = 1e-5
        self.BN_ALPHA = 0.9

        self.generator = Generator()
        self.discriminator = Discriminator()

        if not os.path.exists('images'):
            os.makedirs('images')

        self.disc_losses = []
        self.gen_losses = []

    # Prepare a batch of real and fake data for training the Discriminator.
    def get_batch_for_training_discriminator(self):
        positives = self.all_data[:, np.random.randint(self.all_data.shape[1], size=self.batch_size)]
        negatives = self.generator.get_output(n_outputs=self.batch_size)['a_out']
        pos_labels = np.ones((1, self.batch_size))
        neg_labels = np.zeros((1, self.batch_size))
        train_batch = np.hstack((positives, negatives))
        train_batch_labels = np.hstack((pos_labels, neg_labels))
        shuffle_order = np.random.permutation(train_batch.shape[1])
        train_batch = train_batch[:, shuffle_order]
        train_batch_labels = train_batch_labels[:, shuffle_order]
        return train_batch, train_batch_labels

    # Train the Discriminator over a batch of real and fake data.
    def train_discriminator_over_batch(self, train_batch, train_batch_labels, freeze_weights=True):
        disc_out = self.discriminator.get_output(train_batch)
        preds = disc_out['a_out']
        error = np.average(-train_batch_labels * np.log(preds + self.small_value) - (1 - train_batch_labels) * np.log(1 - preds + self.small_value))

        d_grad = {}
        d_grad['a_out'] = (disc_out['a_out'] - train_batch_labels) / train_batch_labels.shape[1]
        d_grad['z_out'] = disc_out['a_out'] * (1 - disc_out['a_out']) * d_grad['a_out']
        d_grad['W_out'] = np.dot(d_grad['z_out'], disc_out['a_hl2'].T)
        d_grad['b_out'] = np.sum(d_grad['z_out'], axis=1, keepdims=True)
        d_grad['a_hl2'] = np.dot(self.discriminator.d['Wout'].T, d_grad['z_out'])

        d_leaky_relu = np.where(d_grad['a_hl2'] > 0, 1, 0.2)
        d_grad['z_hl2'] = d_grad['a_hl2'] * d_leaky_relu
        d_grad['W_hl2'] = np.dot(d_grad['z_hl2'], disc_out['a_hl1'].T)
        d_grad['b_hl2'] = np.sum(d_grad['z_hl2'], axis=1, keepdims=True)
        d_grad['a_hl1'] = np.dot(self.discriminator.d['Whl2'].T, d_grad['z_hl2'])

        d_leaky_relu = np.where(d_grad['a_hl1'] > 0, 1, 0.2)
        d_grad['z_hl1'] = d_grad['a_hl1'] * d_leaky_relu
        d_grad['W_hl1'] = np.dot(d_grad['z_hl1'], disc_out['input'].T)
        d_grad['b_hl1'] = np.sum(d_grad['z_hl1'], axis=1, keepdims=True)
        d_grad['input'] = np.dot(self.discriminator.d['Whl1'].T, d_grad['z_hl1'])

        if not freeze_weights:
            d_grad = self.discriminator.clip_gradients(d_grad)
            self.discriminator.d.update({
                'Whl1': self.discriminator.d['Whl1'] - self.lr * d_grad['W_hl1'],
                'bhl1': self.discriminator.d['bhl1'] - self.lr * d_grad['b_hl1'],
                'Whl2': self.discriminator.d['Whl2'] - self.lr * d_grad['W_hl2'],
                'bhl2': self.discriminator.d['bhl2'] - self.lr * d_grad['b_hl2'],
                'Wout': self.discriminator.d['Wout'] - self.lr * d_grad['W_out'],
                'bout': self.discriminator.d['bout'] - self.lr * d_grad['b_out']
            })

        return disc_out, d_grad, error

    # Train the Generator over a batch of data.
    def train_generator_over_batch(self, freeze_weights=False):
        train_activations = self.generator.get_output(self.batch_size, train=True)
        train_batch = train_activations['a_out']
        train_batch_labels = np.ones((1, self.batch_size))

        disc_out, grad_from_discriminator, error_disc = self.train_discriminator_over_batch(train_batch, train_batch_labels, freeze_weights=True)
        preds = disc_out['a_out']
        error = np.average(-np.log(preds + self.small_value))

        g_grad = {}
        g_grad['a_out'] = grad_from_discriminator['input']
        g_grad['bn_out'] = (1 - train_activations['a_out'] ** 2) * g_grad['a_out']
        g_grad['Gbn_out'] = np.sum(g_grad['bn_out'] * train_activations['bn_out']['input_normalized'], axis=1, keepdims=True)
        g_grad['Bbn_out'] = np.sum(g_grad['bn_out'], axis=1, keepdims=True)
        g_grad['z_out_norm'] = g_grad['bn_out'] * self.generator.g['Gout']
        g_grad['z_out_var'] = np.sum(g_grad['z_out_norm'] * (train_activations['z_out'] - train_activations['bn_out']['means']), axis=1, keepdims=True)
        std_inv = 1. / np.sqrt(train_activations['bn_out']['variances'] + self.bn_epsilon)
        g_grad['z_out_mean'] = np.sum(g_grad['z_out_norm'] * (-std_inv), axis=1, keepdims=True) + g_grad['z_out_var'] * np.mean(-2. * train_activations['bn_out']['input_normalized'], axis=1, keepdims=True)
        g_grad['z_out'] = (g_grad['z_out_norm'] * std_inv) + (g_grad['z_out_var'] * 2 * train_activations['bn_out']['input_normalized'] / self.batch_size) + (g_grad['z_out_mean'] / self.batch_size)
        g_grad['W_out'] = np.dot(g_grad['z_out'], train_activations['bn_hl3']['output'].T)
        g_grad['b_out'] = np.sum(g_grad['z_out'], axis=1, keepdims=True)
        g_grad['bn_hl3'] = np.dot(self.generator.g['Wout'].T, g_grad['z_out'])

        g_grad['Gbn_hl3'] = np.sum(g_grad['bn_hl3'] * train_activations['bn_hl3']['input_normalized'], axis=1, keepdims=True)
        g_grad['Bbn_hl3'] = np.sum(g_grad['bn_hl3'], axis=1, keepdims=True)
        g_grad['a_hl3_norm'] = g_grad['bn_hl3'] * self.generator.g['Gbn3']
        g_grad['a_hl3_var'] = np.sum(g_grad['a_hl3_norm'] * (train_activations['a_hl3'] - train_activations['bn_hl3']['means']), axis=1, keepdims=True)
        std_inv = 1. / np.sqrt(train_activations['bn_hl3']['variances'] + self.bn_epsilon)
        g_grad['a_hl3_mean'] = np.sum(g_grad['a_hl3_norm'] * (-std_inv), axis=1, keepdims=True) + g_grad['a_hl3_var'] * np.mean(-2. * train_activations['bn_hl3']['input_normalized'], axis=1, keepdims=True)
        g_grad['a_hl3'] = (g_grad['a_hl3_norm'] * std_inv) + (g_grad['a_hl3_var'] * 2 * train_activations['bn_hl3']['input_normalized'] / self.batch_size) + (g_grad['a_hl3_mean'] / self.batch_size)

        d_leaky_relu = np.where(g_grad['a_hl3'] > 0, 1, 0.2)
        g_grad['z_hl3'] = g_grad['a_hl3'] * d_leaky_relu
        g_grad['W_hl3'] = np.dot(g_grad['z_hl3'], train_activations['a_hl2'].T)
        g_grad['b_hl3'] = np.sum(g_grad['z_hl3'], axis=1, keepdims=True)
        g_grad['bn_hl2'] = np.dot(self.generator.g['Whl3'].T, g_grad['z_hl3'])

        g_grad['Gbn_hl2'] = np.sum(g_grad['bn_hl2'] * train_activations['bn_hl2']['input_normalized'], axis=1, keepdims=True)
        g_grad['Bbn_hl2'] = np.sum(g_grad['bn_hl2'], axis=1, keepdims=True)
        g_grad['a_hl2_norm'] = g_grad['bn_hl2'] * self.generator.g['Gbn2']
        g_grad['a_hl2_var'] = np.sum(g_grad['a_hl2_norm'] * (train_activations['a_hl2'] - train_activations['bn_hl2']['means']), axis=1, keepdims=True)
        std_inv = 1. / np.sqrt(train_activations['bn_hl2']['variances'] + self.bn_epsilon)
        g_grad['a_hl2_mean'] = np.sum(g_grad['a_hl2_norm'] * (-std_inv), axis=1, keepdims=True) + g_grad['a_hl2_var'] * np.mean(-2. * train_activations['bn_hl2']['input_normalized'], axis=1, keepdims=True)
        g_grad['a_hl2'] = (g_grad['a_hl2_norm'] * std_inv) + (g_grad['a_hl2_var'] * 2 * train_activations['bn_hl2']['input_normalized'] / self.batch_size) + (g_grad['a_hl2_mean'] / self.batch_size)

        d_leaky_relu = np.where(g_grad['a_hl2'] > 0, 1, 0.2)
        g_grad['z_hl2'] = g_grad['a_hl2'] * d_leaky_relu
        g_grad['W_hl2'] = np.dot(g_grad['z_hl2'], train_activations['a_hl1'].T)
        g_grad['b_hl2'] = np.sum(g_grad['z_hl2'], axis=1, keepdims=True)
        g_grad['bn_hl1'] = np.dot(self.generator.g['Whl2'].T, g_grad['z_hl2'])

        g_grad['Gbn_hl1'] = np.sum(g_grad['bn_hl1'] * train_activations['bn_hl1']['input_normalized'], axis=1, keepdims=True)
        g_grad['Bbn_hl1'] = np.sum(g_grad['bn_hl1'], axis=1, keepdims=True)
        g_grad['a_hl1_norm'] = g_grad['bn_hl1'] * self.generator.g['Gbn1']
        g_grad['a_hl1_var'] = np.sum(g_grad['a_hl1_norm'] * (train_activations['a_hl1'] - train_activations['bn_hl1']['means']), axis=1, keepdims=True)
        std_inv = 1. / np.sqrt(train_activations['bn_hl1']['variances'] + self.bn_epsilon)
        g_grad['a_hl1_mean'] = np.sum(g_grad['a_hl1_norm'] * (-std_inv), axis=1, keepdims=True) + g_grad['a_hl1_var'] * np.mean(-2. * train_activations['bn_hl1']['input_normalized'], axis=1, keepdims=True)
        g_grad['a_hl1'] = (g_grad['a_hl1_norm'] * std_inv) + (g_grad['a_hl1_var'] * 2 * train_activations['bn_hl1']['input_normalized'] / self.batch_size) + (g_grad['a_hl1_mean'] / self.batch_size)

        d_leaky_relu = np.where(g_grad['a_hl1'] > 0, 1, 0.2)
        g_grad['z_hl1'] = g_grad['a_hl1'] * d_leaky_relu
        g_grad['W_hl1'] = np.dot(g_grad['z_hl1'], train_activations['noise_vector'].T)
        g_grad['b_hl1'] = np.sum(g_grad['z_hl1'], axis=1, keepdims=True)

        if not freeze_weights:
            g_grad = self.generator.clip_gradients(g_grad)
            self.generator.g.update({
                'Whl1': self.generator.g['Whl1'] - self.lr * g_grad['W_hl1'],
                'bhl1': self.generator.g['bhl1'] - self.lr * g_grad['b_hl1'],
                'Gbn1': self.generator.g['Gbn1'] - self.lr * g_grad['Gbn_hl1'],
                'Bbn1': self.generator.g['Bbn1'] - self.lr * g_grad['Bbn_hl1'],
                'Whl2': self.generator.g['Whl2'] - self.lr * g_grad['W_hl2'],
                'bhl2': self.generator.g['bhl2'] - self.lr * g_grad['b_hl2'],
                'Gbn2': self.generator.g['Gbn2'] - self.lr * g_grad['Gbn_hl2'],
                'Bbn2': self.generator.g['Bbn2'] - self.lr * g_grad['Bbn_hl2'],
                'Whl3': self.generator.g['Whl3'] - self.lr * g_grad['W_hl3'],
                'bhl3': self.generator.g['bhl3'] - self.lr * g_grad['b_hl3'],
                'Gbn3': self.generator.g['Gbn3'] - self.lr * g_grad['Gbn_hl3'],
                'Bbn3': self.generator.g['Bbn3'] - self.lr * g_grad['Bbn_hl3'],
                'Wout': self.generator.g['Wout'] - self.lr * g_grad['W_out'],
                'bout': self.generator.g['bout'] - self.lr * g_grad['b_out'],
                'Gout': self.generator.g['Gout'] - self.lr * g_grad['Gbn_out'],
                'Bout': self.generator.g['Bout'] - self.lr * g_grad['Bbn_out']
            })

        return train_activations, g_grad, error

    # Train the GAN over a number of epochs.
    def train(self, epochs, save_interval=100):
        for epoch in range(epochs):
            disc_errors = []
            gen_errors = []
            for _ in range(50):
                train_batch, train_batch_labels = self.get_batch_for_training_discriminator()
                _, _, disc_error = self.train_discriminator_over_batch(train_batch, train_batch_labels, freeze_weights=False)
                disc_errors.append(disc_error)
            for _ in range(50):
                _, _, gen_error = self.train_generator_over_batch(freeze_weights=False)
                gen_errors.append(gen_error)

            avg_disc_error = np.mean(disc_errors)
            avg_gen_error = np.mean(gen_errors)
            self.disc_losses.append(avg_disc_error)
            self.gen_losses.append(avg_gen_error)

            print(f"Epoch {epoch + 1}/{epochs} - Discriminator Error: {avg_disc_error:.4f}, Generator Error: {avg_gen_error:.4f}")

            if (epoch + 1) % save_interval == 0:
                self.save_imgs(epoch + 1)

        self.plot_losses()

    # Save generated images to file.
    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (self.latent_dim, r * c))
        gen_imgs = self.generator.get_output(n_outputs=r * c)['a_out']
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[:, cnt].reshape((28, 28)), cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(f"images/mnist_{epoch}.png")
        plt.close()

    # Plot the losses of the Discriminator and Generator.
    def plot_losses(self):
        plt.plot(self.disc_losses, label='Discriminator Loss')
        plt.plot(self.gen_losses, label='Generator Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('losses.png')
        plt.show()


if __name__ == '__main__':
    data_loader = DataLoader(digit=0)
    gan = GAN(data_loader)
    gan.train(epochs=4000, save_interval=100)

