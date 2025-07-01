import tensorflow as tf
from tensorflow.keras import layers

class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            layers.Dense(8 * 8 * 512, use_bias=False, input_shape=(100,)),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Reshape((8, 8, 512)),

            layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'),
        ])

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)
