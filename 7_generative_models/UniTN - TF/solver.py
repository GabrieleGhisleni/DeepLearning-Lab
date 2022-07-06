import tensorflow as tf
import sys
import argparse
import wandb
import os
from model import make_generator_model, make_discriminator_model
import matplotlib.pyplot as plt
import time
import numpy as np
import random


## GAN solver function
class GAN:
    def __init__(self, batch_size, epochs, out_folder='/tmp', ilr=0.0001):
        '''
        Initialization Function, define a few variables that are usful for the training
        :param batch_size:
        :param epochs:
        :param out_folder:
        :param ilr:
        '''
        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_lr = ilr
        self.out_folder = out_folder

    def train(self, train_dataset):
        '''
        Training function receiving the training dataset that must be compliant with the model
        :param train_dataset:
        :return:
        '''

        ## instantiating the generator and the discriminator from model.py
        generator = make_generator_model()
        discriminator = make_discriminator_model()

        ## defining the losses for the generator and the discriminator
        def generator_loss(fake_output):
            return tf.reduce_mean(tf.square(tf.ones_like(fake_output) - fake_output))

        def discriminator_loss(real_output, fake_output):
            real_loss = tf.reduce_mean(tf.square(tf.ones_like(real_output) - real_output))
            fake_loss = tf.reduce_mean(tf.square(tf.zeros_like(fake_output) - fake_output))
            total_loss = real_loss + fake_loss
            return total_loss

        ## Defining the two optimizers, one for the generator and one for the discriminator
        generator_optimizer = tf.keras.optimizers.Adam(self.initial_lr)
        discriminator_optimizer = tf.keras.optimizers.Adam(self.initial_lr)

        ## Defining the input shape of the generator (noise vector) and how many imgs we shall generate at every epoch.
        ## The seed is needed to browse output from the same input for test
        noise_dim = 100
        num_examples_to_generate = 16
        seed = tf.random.normal([num_examples_to_generate, noise_dim])

        @tf.function
        def train_step(images):
            ## Generate input noise at each batch
            noise = tf.random.normal([self.batch_size, noise_dim])

            ## Compute forward pass and gradients using losses
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
            return gen_loss, disc_loss

        def generate_and_save_images(model, epoch, test_input):
            '''
            A function to generate images as output
            :param model:
            :param epoch:
            :param test_input:
            :return:
            '''

            # Notice `training` is set to False.
            # This is so all layers run in inference mode (batchnorm).
            predictions = model(test_input, training=False)

            fig = plt.figure(figsize=(4, 4))
            imlist = []
            for i in range(predictions.shape[0]):
                plt.subplot(4, 4, i + 1)
                image = predictions[i, :, :, 0] * 127.5 + 127.5
                myimage = wandb.Image(image)
                imlist.append(myimage)
                plt.imshow(image, cmap='gray')
                plt.axis('off')
            wandb.log({'image': imlist, 'epoch': epoch + 1})
            plt.savefig(os.path.join(self.out_folder, 'image_at_epoch_{:04d}.png'.format(epoch)))

        ## Training loop
        global_step = 0
        for epoch in range(self.epochs):
            start = time.time()

            for image_batch, label_batch in train_dataset:
                global_step += 1
                gen_loss, disc_loss = train_step(image_batch)
                if global_step % 100  == 0:
                    wandb.log({'generator_loss': gen_loss, 'discriminator_loss': disc_loss})

            # Produce images for the GIF as you go
            generate_and_save_images(generator,
                                     epoch + 1,
                                     seed)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        ## Generate after the final epoch
        generate_and_save_images(generator,
                                 self.epochs,
                                 seed)

