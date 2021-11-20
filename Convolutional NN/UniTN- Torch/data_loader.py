import os
import glob
import cv2
import numpy as np
import random
import datetime
import pickle
from torch.utils.data import Dataset


def load_stl(data_path = 'stl10', norm=True):
    def normalize(data):
        data = (data / 255) - 0.5
        return data

    def load_data(mypath):
        data = np.zeros((0, 50, 50, 3), dtype='uint8')
        labels = np.zeros((0,))
        for i, cla in enumerate(mypath):
            filelist = glob.glob(os.path.join(cla, '*.jpg'))
            tmp_data = np.empty((len(filelist), 50, 50, 3), dtype='uint8')
            tmp_labels = np.ones((len(filelist),)) * i

            for j, path in enumerate(filelist):
                image = cv2.imread(path)
                resized = cv2.resize(image, (50,50))
                tmp_data[j, :] = resized
            data = np.concatenate((data, tmp_data))
            labels = np.concatenate((labels, tmp_labels))
        return data, labels

    train_path = glob.glob(os.path.join(data_path, 'train', '*'))
    train_path.sort()
    test_path = glob.glob(os.path.join(data_path, 'test', '*'))
    test_path.sort()
    training_data, training_labels = load_data(train_path)
    test_data, test_labels = load_data(test_path)
    perm = np.arange(test_data.shape[0])
    random.shuffle(perm)
    perm = perm[:1000]
    test_data = test_data[perm, :]
    test_labels = test_labels[perm]
    if norm:
        training_data = normalize(training_data)
        test_data = normalize(test_data)
    return training_data, training_labels, test_data, test_labels
