import tensorflow as tf
import sys
import argparse
import wandb
import os
from model import make_generator_model, make_discriminator_model
from solver import GAN as GANsolver
import matplotlib.pyplot as plt
import time
import scipy.io
import numpy as np


def GAN(batch_size, epochs):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    BUFFER_SIZE = 60000

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(
        batch_size)

    solver = GANsolver(batch_size=batch_size, epochs=epochs, ilr=0.0001)
    solver.train(train_dataset)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lecture on autoencoder and image classification.')
    parser.add_argument('-mode', type=str, default='gan', help='training or test')
    parser.add_argument('-n', type=str, default='test')
    parser.add_argument('-e', type=int, default=30)
    parser.add_argument('-bs', type=int, default=64)
    parser.add_argument('-i', type=str, default=None)
    parser.add_argument('-wandb', type=str, default='True', help='Log on WandB (default = True)')
    args = parser.parse_args()

    # trigger or untrigger WandB
    if args.wandb == 'False' or args.mode == 'deploy':
        os.environ['WANDB_MODE'] = 'dryrun'

    # 1. Start a W&B run
    wandb.init(project='aml-domain_adaptation', entity='unitn-mhug', group=args.mode, name=args.n)
    wandb.config.epochs = args.e
    wandb.config.batch_size = args.bs


    GAN(batch_size=args.bs, epochs=args.e)