#!/usr/bin/env python3
"""
DCGAN Variant Comparison Script

This script trains and compares multiple variants of Deep Convolutional GAN (DCGAN) models
on the MNIST dataset. Each variant can differ in architecture or hyperparameters.
Generated images are saved for visual inspection, and training metrics can be tracked via Weights and Biases.

Author: Donny / Nova
Date: 2025
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback

def make_generator_model():
    """
    Create the generator model for a DCGAN.

    Returns:
        tf.keras.Sequential: Generator model.
    """
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

def make_discriminator_model():
    """
    Create the discriminator model for a DCGAN.

    Returns:
        tf.keras.Sequential: Discriminator model.
    """
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

def discriminator_loss(real_output, fake_output):
    """
    Calculate the discriminator loss.

    Args:
        real_output: Discriminator output on real images.
        fake_output: Discriminator output on generated images.

    Returns:
        tf.Tensor: Total loss.
    """
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    """
    Calculate the generator loss.

    Args:
        fake_output: Discriminator output on generated images.

    Returns:
        tf.Tensor: Generator loss.
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(images, generator):
    """
    Run a single training step.

    Args:
        images: A batch of real images.
        generator: Generator model.

    Returns:
        None
    """
    noise = tf.random.normal([config.batch_size, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def generate_and_save_images(model, epoch, test_input, variant):
    """
    Generate and save images produced by the generator.

    Args:
        model: Generator model.
        epoch: Current epoch number.
        test_input: Noise vector input.
        variant: Variant index or label.

    Returns:
        None
    """
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(f'image_at_epoch_{epoch:04d}_variant_{variant}.png')
    plt.show()

def train(dataset, epochs, generator, discriminator, save_interval, variant):
    """
    Run the training loop for a single DCGAN variant.

    Args:
        dataset: Preprocessed training dataset.
        epochs: Number of training epochs.
        generator: Generator model.
        discriminator: Discriminator model.
        save_interval: How often to save generated images.
        variant: Identifier for the model variant.
    """
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch, generator)
        if (epoch + 1) % save_interval == 0:
            noise = tf.random.normal([16, 100])
            generate_and_save_images(generator, epoch + 1, noise, variant)

save_interval = 10

wandb.init(project="dcgan_mnist")
config = wandb.config
config.epochs = 80
config.batch_size = 256

(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE = 60000
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(config.batch_size)

generator_variant_1 = make_generator_model()
discriminator_variant_1 = make_discriminator_model()

generator_variant_2 = make_generator_model()
discriminator_variant_2 = make_discriminator_model()

for variant, (generator, discriminator) in enumerate([
    (generator_variant_1, discriminator_variant_1),
    (generator_variant_2, discriminator_variant_2)
]):
    wandb.init(project=f"dcgan_mnist_variant_{variant}")
    train(train_dataset, config.epochs, generator, discriminator, save_interval, variant)
