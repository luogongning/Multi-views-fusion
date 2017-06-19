from __future__ import print_function

import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os

#from modelvgg19 import get_model
#from utils import preprocess, rotation_augmentation, shift_augmentation


def load_train_data():
    """
    Load training data from .npy files.
    """
    debug_folder = os.path.join('D:', 'data', 'data')
    X = np.load('data/X_train_ALL.npy')
    y = np.load('data/y_train.npy')
    print('CRPS(train) = {0}'.format(y.shape[0]))
    print('CRPS(train) = {0}'.format(X.shape[0]))
    print('CRPS(train) = {0}'.format(X.shape[1]))
    print('CRPS(train) = {0}'.format(X.shape[2]))
    print('CRPS(train) = {0}'.format(X.shape[3]))
    a=X.shape[0]
    b=X.shape[1]
    for i in range(a):
        for j in range(b):
            crop_img=X[i,j, :, :]
            cv2.imwrite(os.path.join(debug_folder, str(i) + '_' + str(j)+'.jpg'), crop_img)

    #X = X.astype(np.float32)
    #X /= 255

    #seed = np.random.randint(1, 10e6)
    #np.random.seed(seed)
    #np.random.shuffle(X)
    #np.random.seed(seed)
    #np.random.shuffle(y)

    return X, y

load_train_data()