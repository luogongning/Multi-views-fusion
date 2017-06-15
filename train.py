from __future__ import print_function

import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from vgg19_2 import VGG19
from utils import preprocess, rotation_augmentation, shift_augmentation


def load_train_data():
    """
    Load training data from .npy files.
    """
    X = np.load('data/X_train_ALL.npy')
    y = np.load('data/y_train.npy')

    X = X.astype(np.float32)
    X /= 255

    return X, y


def split_data(X, y, split_ratio=0.6):
    """
    Split data into training and testing.

    :param X: X
    :param y: y
    :param split_ratio: split ratio for train and test data
    """
    #split = int(X.shape[0] * split_ratio)
    #the input dimensions can be control through modifing the second axis.
    X_train= X[:, 1:4, :, :]
    y_train=y
    """
    Load validation data from .npy files.
    """
    X_test_R = np.load('data/X_validation_ALL.npy')
    X_test_R = X_test_R.astype(np.float32)
    X_test_R /= 255
    X_test_R=preprocess(X_test_R)
    X_test = X_test_R[:, 1:4, :, :]
    #y_test = int(y_test_R*0.2)
    
    y_test_R = np.load('data/y_validation.npy')
    y_test=y_test_R

    return X_train, y_train, X_test, y_test


def train():
    """
    Training systole and diastole models.
    """
    print('Loading and compiling models...')
    #model_systole = get_model()
    model_diastole = VGG19()

    print('Loading training data...')
    X, y = load_train_data()

    print('Pre-processing images...')
    X = preprocess(X)

    # split to training and test
    X_train, y_train, X_test, y_test = split_data(X, y, split_ratio=0.6)
    #loss=0
    nb_iter = 1000
    epochs_per_iter = 1
    batch_size = 64
    #calc_crps = 1  # calculate CRPS every n-th iteration (set to 0 if CRPS estimation is not needed)

    # remember min val. losses (best iterations), used as sigmas for submission
    min_val_loss_diastole = sys.float_info.max

    print('-'*50)
    print('Training...')
    print('-'*50)

    for i in range(nb_iter):
        print('-'*50)
        print('Iteration {0}/{1}'.format(i + 1, nb_iter))
        print('-'*50)
        
        print('Augmenting images - rotations')
        X_train_aug = rotation_augmentation(X_train, 10)
        print('Augmenting images - shifts')
        X_train_aug = shift_augmentation(X_train_aug, 0.1, 0.1)

        print('Fitting diastole model...')
        hist_diastole = model_diastole.fit(X_train_aug, y_train, shuffle=True, nb_epoch=epochs_per_iter,
                                           batch_size=batch_size, validation_data=(X_test, y_test))
		
        loss_diastole = hist_diastole.history['loss'][-1]
        
        val_loss_diastole = hist_diastole.history['val_loss'][-1]
        with open('diastole_loss_processing.txt', mode='a') as f1:
            f1.write(str('loss='))
            f1.write(str(loss_diastole))
            f1.write(' ')
            f1.write(str('val_loss='))
            f1.write(str(val_loss_diastole))
            f1.write('\n')
            f1.close()
        print('Saving weights...')
        # save weights so they can be loaded later
        model_diastole.save_weights('weights_diastole.hdf5', overwrite=True)

        if val_loss_diastole < min_val_loss_diastole:
            min_val_loss_diastole = val_loss_diastole
            model_diastole.save_weights('weights_diastole_best.hdf5', overwrite=True)

        # save best (lowest) val losses in file (to be later used for generating submission)
        with open('val_loss_ED.txt', mode='w+') as f:
            #f.write(str(min_val_loss_systole))
            #f.write('\n')
            f.write(str(min_val_loss_diastole))
            f.close()


train()
