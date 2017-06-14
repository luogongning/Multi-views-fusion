from __future__ import print_function

import csv
import numpy as np

from model import get_model_ALL_ES, get_model_ALL_ED
from utils import preprocess


def load_validation_data():
    """
    Load validation data from .npy files.
    """
    X = np.load('data/X_test_ES_ALL.npy')
    X2 = np.load('data/X_test_ED_ALL.npy')
    ids = np.load('data/ids_test.npy')

    X = X.astype(np.float32)
    X /= 255
    X2 = X2.astype(np.float32)
    X2 /= 255

    return X, X2, ids

print('Loading and compiling models...')
model_systole = get_model_ALL_ES()
model_diastole = get_model_ALL_ED()

print('Loading models weights...')
model_systole.load_weights('weights_systole_best.hdf5')
model_diastole.load_weights('weights_diastole_best.hdf5')

    # load val losses to use as sigmas for CDF
with open('val_loss.txt', mode='r') as f:
    val_loss_systole = float(f.readline())
    val_loss_diastole = float(f.readline())

print('Loading test data...')
X, X2, ids = load_validation_data()

print('Pre-processing images...')
X = preprocess(X)
X2 = preprocess(X2)

batch_size = 64
print('Predicting on test data...')
pred_systole = model_systole.predict(X, batch_size=batch_size, verbose=1)
pred_diastole = model_diastole.predict(X2, batch_size=batch_size, verbose=1)

    # write to submission file
print('Writing submission to file...')
fi = csv.reader(open('data/sample_submission_test.csv'))
f = open('submission_test.csv', 'w')
fo = csv.writer(f, lineterminator='\n')
#define the title of table for every column
fo.writerow(fi.next())
count=0
for id in range(len(ids)):
    count+=1
    idx = ids[id]
    #key, target = idx.split('_')
    #key = int(key)
    out = [idx]
    out.extend(pred_systole[id])
    out.extend(pred_diastole[id])
    fo.writerow(out)
f.close()

print('Done.count={0}'.format(count))

