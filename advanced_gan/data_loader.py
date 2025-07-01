import tensorflow as tf
import os

def get_dataset(data_dir="data", image_size=(64, 64), batch_size=128):
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode=None,
        image_size=image_size,
        batch_size=batch_size
    )
    dataset = dataset.map(lambda x: (x - 127.5) / 127.5)  # Normalize to [-1, 1]
    return dataset
