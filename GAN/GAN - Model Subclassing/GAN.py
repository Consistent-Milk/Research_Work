import tensorflow as tf
from tensorflow import keras

class GAN(keras.Model):

    def __init__(self, discriminator, generator, latent_dim):
        super.__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_loss_metric = keras.metrics.Mean()