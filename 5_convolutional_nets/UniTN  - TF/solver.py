import tensorflow as tf
import random
import numpy as np


class ImageClassifier():
    def __init__(self, batch_size, epochs, ilr=0.0001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.initial_learning_rate = ilr

    def train(self, training_data, training_labels, test_data, test_labels, model):
        supervised_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.initial_learning_rate)

        @tf.function
        def train_step(data, labels):
            with tf.GradientTape() as tape:
                logits, preds = model(data, training=True)
                loss = supervised_loss(y_true=labels, y_pred=logits)
            trainable_vars = model.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(gradients, trainable_vars))
            eq = tf.equal(labels, tf.argmax(preds, -1))
            accuracy = tf.reduce_mean(tf.cast(eq, tf.float32)) * 100
            return loss, accuracy

        @tf.function
        def test_step(data, labels):
            logits, preds = model(data, training=False)
            loss = supervised_loss(y_true=labels, y_pred=logits)
            return loss, logits, preds

        global_step = 0
        best_accuracy = 0.0
        for e in range(self.epochs):

            ## Shuffling training set
            perm = np.arange(len(training_labels))
            random.shuffle(perm)
            training_data = training_data[perm]
            training_labels = training_labels[perm]

            ## Iteration
            for i in range(0, len(training_labels), self.batch_size):
                data = training_data[i:i+self.batch_size, :]
                labels = training_labels[i:i+self.batch_size, ].astype('int64')
                global_step += 1  # len(labels)
                batch_loss, batch_accuracy = train_step(data, labels)
                if global_step % 50 == 0:
                    print('[{0}-{1:03}] loss: {2:0.05}, batch_accuracy: {3:0.03}'.format(
                        e + 1, global_step,
                        batch_loss.numpy(),
                        batch_accuracy.numpy()))
                if global_step == 1:
                    print('number of model parameters {}'.format(model.count_params()))

            # Test the whole test dataset
            test_preds = tf.zeros((0,), dtype=tf.int64)
            total_loss = list()
            for i in range(0, len(test_labels), self.batch_size):
                data = test_data[i:i+self.batch_size, :]
                labels = test_labels[i:i+self.batch_size, ].astype('int64')
                batch_loss, _, preds = test_step(data, labels)
                batch_preds = tf.argmax(preds, -1)
                test_preds = tf.concat([test_preds, batch_preds], axis=0)
                total_loss.append(batch_loss)
            loss = sum(total_loss)/len(total_loss)
            eq = tf.equal(test_labels, test_preds)
            test_accuracy = tf.reduce_mean(tf.cast(eq, tf.float32)) * 100
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
            print('End of Epoch {0}/{1:03} -> loss: {2:0.05}, test accuracy: {3:0.03} - best accuracy: {4:0.03}'.format(
                        e + 1, self.epochs,
                        loss.numpy(),
                        test_accuracy.numpy(),
                        best_accuracy))


        
