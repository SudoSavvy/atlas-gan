not empty
# DCGAN on MNIST

This project implements and experiments with a Deep Convolutional GAN on the MNIST dataset.

## Structure

- `models/`: Generator, Discriminator, and GAN builder
- `utils/`: Data preprocessing
- `train.py`: Training + WandB logging

## Experiments

All experiments tracked with [Weights & Biases](https://wandb.ai/)

Run baseline:
```bash
python3 train.py
