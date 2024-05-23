import numpy as np
import os
import pickle
import wfdb
import csv
from tqdm import tqdm
from sklearn.model_selection import train_test_split

data_folder = 'physionet2017/raw'
aliases = {'train' : 'training2017', 'test' : 'validation2017'}

sampling_fs = 300 # Hz
window_len = 1500 # samples
label_map = {'N':0, 'A':1, 'O':2, '~':3}

signal_min = float('inf')
signal_max = float('-inf')
signal_lens = []
# read training and testing datasets into two lists
def get_X_y(alias):
    global signal_min, signal_max
    X = []
    y = []
    basepath = f'{os.getcwd()}/{data_folder}/{alias}'
    # First get list of file names from the data folder
    file_names = [file_name.split('.hea')[0] for file_name in os.listdir(basepath) if '.hea' in file_name]
    file_names.sort()
    diagnoses = []
    with open(os.path.join(basepath, 'REFERENCE.csv'), 'r') as f:
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            diagnoses.append(line[1])

     # Process each file by dividing up time series into contiguous windows
    for file_name, diagnosis in tqdm(zip(file_names, diagnoses)):
        signal, _ = wfdb.rdsamp(os.path.join(basepath, file_name))
        signal_min = min(signal_min, signal.min())
        signal_max = max(signal_max, signal.max())
        X.append(signal)
        y.append(label_map[diagnosis])
        signal_lens.append(len(signal))
    return X, y

# returns X of dimension: n_samples x n_channels x window_len
#         y ...         : n_samples x window_len
# Makes all patient's signal of same length, equal to the shortest one
def cut_signal(X, y):
    min_len = min(signal_lens)
    X = np.array([arr[:min_len] for arr in X])
    X = np.swapaxes(X, 1, 2)
    X = (X - 0) / (signal_max - signal_min) # normalize
    y = np.expand_dims(np.array(y), axis=1)
    y = np.repeat(y, repeats=min_len, axis=1)
    print(X.shape, y.shape)
    return X, y

X_train, y_train = get_X_y(aliases['train'])
X_test, y_test = get_X_y(aliases['test'])

X_train, y_train = cut_signal(X_train, y_train)
X_test, y_test = cut_signal(X_test, y_test)

# assumes we run script from the inner data folder
output_dir = os.path.join(os.getcwd(), 'physionet2017', 'processed')

with open(os.path.join(output_dir, 'x_train.pkl'), 'wb') as f:
    pickle.dump(X_train, f)
with open(os.path.join(output_dir, 'state_train.pkl'), 'wb') as f:
    pickle.dump(y_train, f)
with open(os.path.join(output_dir, 'x_test.pkl'), 'wb') as f:
    pickle.dump(X_test, f)
with open(os.path.join(output_dir, 'state_test.pkl'), 'wb') as f:
    pickle.dump(y_test, f)