import tensorflow as tf
from tensorflow.keras import layers

class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                          input_shape=[64, 64, 3]),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),

            layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),

            layers.Flatten(),
            layers.Dense(1),
        ])

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)
