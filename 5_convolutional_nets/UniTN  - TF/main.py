import tensorflow as tf
import sys
from data_loader import load_stl
from model import Densenet, MobileNet, ResNet
from solver import ImageClassifier



def main():
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
        sys.exit(1)
    training_data, training_labels, test_data, test_labels = load_stl()
    model = ResNet()
    ic = ImageClassifier(batch_size=32, epochs=100)
    ic.train(training_data, training_labels, test_data, test_labels, model)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

