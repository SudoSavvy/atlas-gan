#!/usr/bin/env python3
"""
Train script for advanced GAN experiments.
"""

from data_loader import get_dataset
from models.generator import Generator
from models.discriminator import Discriminator
from train_utils import train_gan
import tensorflow as tf

if __name__ == "__main__":
    dataset = get_dataset("advanced_gan/data/celeba/img_align_celeba")
    
    generator = Generator()
    discriminator = Discriminator()
    
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    
    train_gan(
        dataset,
        generator,
        discriminator,
        gen_optimizer,
        disc_optimizer,
        epochs=100,
        batch_size=128
    )
