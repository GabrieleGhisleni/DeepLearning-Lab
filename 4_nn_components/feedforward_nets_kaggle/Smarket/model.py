from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
import tensorflow as tf

class MarketClassifier(tf.keras.Model):
    def __init__(self, input_shape, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = [input_shape]
        self.layer_1 = Dense(16, activation= tf.nn.relu, input_shape=[input_shape[1]])
        self.layer_2 = Dense(32, activation= tf.nn.relu)
        self.layer_3 = Dense(16, activation= tf.nn.relu)
        self.out = Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.out(x)
        return x
