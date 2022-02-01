import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer


class Densenet(Model):
    def __init__(self):
        super(Densenet, self).__init__(trainable=True)
        self.backbone = tf.keras.applications.DenseNet121(include_top=False, input_shape=[96, 96, 3])
        self.final_conv = tf.keras.layers.Conv2D(filters=10, kernel_size=[3, 3], padding='valid')
        self.final_flatten = tf.keras.layers.Flatten()

    def call(self, inputs, training=None, mask=None):
        x = self.backbone(inputs)
        x = self.final_conv(x)
        x = self.final_flatten(x)
        return x, tf.keras.activations.softmax(x)


class MobileNet(Model):
    def __init__(self):
        super(MobileNet, self).__init__(trainable=True)
        self.backbone = tf.keras.applications.MobileNetV2(include_top=False, input_shape=[96, 96, 3])
        self.final_conv = tf.keras.layers.Conv2D(filters=10, kernel_size=[3, 3], padding='valid')
        self.final_flatten = tf.keras.layers.Flatten()

    def call(self, inputs, training=None, mask=None):
        x = self.backbone(inputs)
        x = self.final_conv(x)
        x = self.final_flatten(x)
        return x, tf.keras.activations.softmax(x)

class ResNet(Model):
    def __init__(self):
        super(ResNet, self).__init__(trainable=True)
        self.backbone = tf.keras.applications.ResNet50(include_top=False, input_shape=[96, 96, 3])
        self.final_conv = tf.keras.layers.Conv2D(filters=10, kernel_size=[3, 3], padding='valid')
        self.final_flatten = tf.keras.layers.Flatten()

    def call(self, inputs, training=None, mask=None):
        x = self.backbone(inputs)
        x = self.final_conv(x)
        x = self.final_flatten(x)
        return x, tf.keras.activations.softmax(x)