from __future__ import print_function

import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from model import get_model_ALL_ES
from utils import preprocess, rotation_augmentation, shift_augmentation


def load_train_data():
    """
    Load training data from .npy files.
    """
    X = np.load('data/X_train_ES_ALL.npy')
    y = np.load('data/y_train.npy')

    X = X.astype(np.float32)
    X /= 255

    #seed = np.random.randint(1, 10e6)
    #np.random.seed(seed)
    #np.random.shuffle(X)
    #np.random.seed(seed)
    #np.random.shuffle(y)

    return X, y


def split_data(X, y, split_ratio=0.2):
    """
    Split data into training and testing.

    :param X: X
    :param y: y
    :param split_ratio: split ratio for train and test data
    """
    split = X.shape[0] * split_ratio
    X_test = X[:split, 2:3, :, :]
    y_test = y[:split, :]
    X_train = X[split:, 2:3, :, :]
    y_train = y[split:, :]

    return X_train, y_train, X_test, y_test


def train():
    """
    Training systole and diastole models.
    """
    print('Loading and compiling models...')
    model_systole = get_model_ALL_ES()
   #model_diastole = get_model()

    print('Loading training data...')
    X, y = load_train_data()

    print('Pre-processing images...')
    X = preprocess(X)

    # split to training and test
    #X_train, y_train, X_test, y_test = split_data(X, y, split_ratio=0.2)

    nb_iter = 150
    epochs_per_iter = 1
    batch_size = 32
    #calc_crps = 1  # calculate CRPS every n-th iteration (set to 0 if CRPS estimation is not needed)

    # remember min val. losses (best iterations), used as sigmas for submission
    min_val_loss_systole = sys.float_info.max
    #min_val_loss_diastole = sys.float_info.max

    print('-'*50)
    print('Training...')
    print('-'*50)

    for i in range(nb_iter):
        print('-'*50)
        print('Iteration {0}/{1}'.format(i + 1, nb_iter))
        print('-'*50)
        seed = np.random.randint(1, 10e6)
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
        X_train, y_train, X_test, y_test = split_data(X, y, split_ratio=0.2)
        print('Augmenting images - rotations')
        X_train_aug = rotation_augmentation(X_train, 10)
        print('Augmenting images - shifts')
        X_train_aug = shift_augmentation(X_train_aug, 0.1, 0.1)

        print('Fitting systole model...')
        hist_systole = model_systole.fit(X_train_aug, y_train[:, 0], shuffle=True, nb_epoch=epochs_per_iter,
                                         batch_size=batch_size, validation_data=(X_test, y_test[:, 0]))
		#print(hist_systole.history)
        #print('Fitting diastole model...')
        #hist_diastole = model_diastole.fit(X_train_aug, y_train[:, 1], shuffle=True, nb_epoch=epochs_per_iter,
                                           #batch_size=batch_size, validation_data=(X_test, y_test[:, 1]))
		#print(hist_diastole.history)
		#f=open('diastole.txt','w')
		#f.write(hist_diastole.history)
		#f.close()
        # sigmas for predicted data, actually loss function values (RMSE)
        loss_systole = hist_systole.history['loss'][-1]
        #loss_diastole = hist_diastole.history['loss'][-1]
        val_loss_systole = hist_systole.history['val_loss'][-1]
        #val_loss_diastole = hist_diastole.history['val_loss'][-1]
        with open('systole_loss_processing.txt', mode='a') as f1:
            f1.write(str(loss_systole))
            f1.write('\n')
            #f1.write(str(loss_diastole))
            #f1.write('\n')
            f1.write(str(val_loss_systole))
            f1.write('\n')
            #f1.write(str(val_loss_diastole))
            #f1.write('\n')
            f1.close()
        #if calc_crps > 0 and i % calc_crps == 0:
            #print('Evaluating CRPS...')
            #pred_systole = model_systole.predict(X_train, batch_size=batch_size, verbose=1)
            #pred_diastole = model_diastole.predict(X_train, batch_size=batch_size, verbose=1)
            #val_pred_systole = model_systole.predict(X_test, batch_size=batch_size, verbose=1)
            #val_pred_diastole = model_diastole.predict(X_test, batch_size=batch_size, verbose=1)

            # CDF for train and test data (actually a step function)
            #cdf_train = real_to_cdf(np.concatenate((y_train[:, 0], y_train[:, 1])))
            #cdf_test = real_to_cdf(np.concatenate((y_test[:, 0], y_test[:, 1])))

            # CDF for predicted data
            #cdf_pred_systole = real_to_cdf(pred_systole, loss_systole)
            #cdf_pred_diastole = real_to_cdf(pred_diastole, loss_diastole)
            #cdf_val_pred_systole = real_to_cdf(val_pred_systole, val_loss_systole)
            #cdf_val_pred_diastole = real_to_cdf(val_pred_diastole, val_loss_diastole)

            # evaluate CRPS on training data
            #crps_train = crps(cdf_train, np.concatenate((cdf_pred_systole, cdf_pred_diastole)))
            #print('CRPS(train) = {0}'.format(crps_train))

            # evaluate CRPS on test data
            #crps_test = crps(cdf_test, np.concatenate((cdf_val_pred_systole, cdf_val_pred_diastole)))
            #print('CRPS(test) = {0}'.format(crps_test))
            #with open('crps_processing.txt', mode='a') as f2:
                #f2.write(str(crps_train))
                #f2.write('\n')
                #f2.write(str(crps_test))
                #f2.write('\n')
                #f2.close()
        print('Saving weights...')
        # save weights so they can be loaded later
        model_systole.save_weights('weights_systole.hdf5', overwrite=True)
        #model_diastole.save_weights('weights_diastole.hdf5', overwrite=True)

        # for best (lowest) val losses, save weights
        if val_loss_systole < min_val_loss_systole:
            min_val_loss_systole = val_loss_systole
            model_systole.save_weights('weights_systole_best.hdf5', overwrite=True)

        #if val_loss_diastole < min_val_loss_diastole:
            #min_val_loss_diastole = val_loss_diastole
            #model_diastole.save_weights('weights_diastole_best.hdf5', overwrite=True)

        # save best (lowest) val losses in file (to be later used for generating submission)
        with open('val_loss_ES.txt', mode='w+') as f:
            f.write(str(min_val_loss_systole))
            #f.write('\n')
            #f.write(str(min_val_loss_diastole))
            f.close()


train()
