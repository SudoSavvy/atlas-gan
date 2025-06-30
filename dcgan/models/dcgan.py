#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
# Removed wandb import as you requested skipping wandb

def build_generator_network():
    """
    Construct the generator network that creates images from random noise vectors.

    Returns:
        tf.keras.Model: Generator neural network.
    """
    gen_model = tf.keras.Sequential()
    gen_model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    gen_model.add(layers.BatchNormalization())
    gen_model.add(layers.LeakyReLU())

    gen_model.add(layers.Reshape((7, 7, 256)))

    gen_model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1),
                                         padding='same', use_bias=False))
    gen_model.add(layers.BatchNormalization())
    gen_model.add(layers.LeakyReLU())

    gen_model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2),
                                         padding='same', use_bias=False))
    gen_model.add(layers.BatchNormalization())
    gen_model.add(layers.LeakyReLU())

    gen_model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2),
                                         padding='same', use_bias=False,
                                         activation='tanh'))
    return gen_model

def build_discriminator_network():
    """
    Build the discriminator network that classifies real vs fake images.

    Returns:
        tf.keras.Model: Discriminator neural network.
    """
    disc_model = tf.keras.Sequential()
    disc_model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                 input_shape=[28, 28, 1]))
    disc_model.add(layers.LeakyReLU())
    disc_model.add(layers.Dropout(0.3))

    disc_model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    disc_model.add(layers.LeakyReLU())
    disc_model.add(layers.Dropout(0.3))

    disc_model.add(layers.Flatten())
    disc_model.add(layers.Dense(1))
    return disc_model

cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def calculate_discriminator_loss(real_logits, fake_logits):
    """
    Compute loss for discriminator by comparing real and generated image logits.

    Args:
        real_logits (tf.Tensor): Discriminator output on real images.
        fake_logits (tf.Tensor): Discriminator output on fake images.

    Returns:
        tf.Tensor: Total discriminator loss.
    """
    real_loss = cross_entropy_loss(tf.ones_like(real_logits), real_logits)
    fake_loss = cross_entropy_loss(tf.zeros_like(fake_logits), fake_logits)
    return real_loss + fake_loss

def calculate_generator_loss(fake_logits):
    """
    Compute loss for generator based on discriminator's output on fake images.

    Args:
        fake_logits (tf.Tensor): Discriminator output on generated images.

    Returns:
        tf.Tensor: Generator loss.
    """
    return cross_entropy_loss(tf.ones_like(fake_logits), fake_logits)

generator_opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
discriminator_opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

@tf.function
def run_training_step(real_images, generator_net, discriminator_net, batch_size):
    """
    Perform one training step updating both generator and discriminator.

    Args:
        real_images (tf.Tensor): Batch of real images.
        generator_net (tf.keras.Model): Generator network.
        discriminator_net (tf.keras.Model): Discriminator network.
        batch_size (int): Batch size for noise input.
    """
    noise_vector = tf.random.normal([batch_size, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator_net(noise_vector, training=True)

        real_output = discriminator_net(real_images, training=True)
        fake_output = discriminator_net(generated_images, training=True)

        gen_loss = calculate_generator_loss(fake_output)
        disc_loss = calculate_discriminator_loss(real_output, fake_output)

    gradients_gen = gen_tape.gradient(gen_loss, generator_net.trainable_variables)
    gradients_disc = disc_tape.gradient(disc_loss, discriminator_net.trainable_variables)

    generator_opt.apply_gradients(zip(gradients_gen, generator_net.trainable_variables))
    discriminator_opt.apply_gradients(zip(gradients_disc, discriminator_net.trainable_variables))

def save_generated_images(generator_net, epoch_num, noise_input):
    """
    Generate images from the generator and save a grid of results.

    Args:
        generator_net (tf.keras.Model): Generator model.
        epoch_num (int): Current epoch number.
        noise_input (tf.Tensor): Noise vectors for image generation.
    """
    predictions = generator_net(noise_input, training=False)
    plt.figure(figsize=(4, 4))

    for idx in range(predictions.shape[0]):
        plt.subplot(4, 4, idx + 1)
        img = predictions[idx, :, :, 0] * 127.5 + 127.5
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    plt.savefig(f'generated_epoch_{epoch_num:04d}.png')
    plt.close()

def train_gan_model(dataset, total_epochs, generator_net, discriminator_net,
                    batch_size, save_every):
    """
    Train the GAN model for a specified number of epochs.

    Args:
        dataset (tf.data.Dataset): Dataset of real images.
        total_epochs (int): Number of epochs to train.
        generator_net (tf.keras.Model): Generator model.
        discriminator_net (tf.keras.Model): Discriminator model.
        batch_size (int): Batch size.
        save_every (int): Interval epochs to save generated images.
    """
    for epoch in range(total_epochs):
        for batch_images in dataset:
            run_training_step(batch_images, generator_net, discriminator_net, batch_size)

        if (epoch + 1) % save_every == 0:
            noise = tf.random.normal([16, 100])
            save_generated_images(generator_net, epoch + 1, noise)
            print(f"Epoch {epoch + 1} completed and images saved.")

# Configurable parameters
EPOCHS = 80
BATCH_SIZE = 256
SAVE_INTERVAL = 10

# Load and preprocess MNIST dataset
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]

BUFFER_SIZE = 60000
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Initialize models
generator_network = build_generator_network()
discriminator_network = build_discriminator_network()

# Start training the GAN
train_gan_model(train_dataset, EPOCHS, generator_network, discriminator_network,
                BATCH_SIZE, SAVE_INTERVAL)
