import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical


def load_data(filepath):
    with h5py.File(filepath, 'r') as f:
        images = np.array(f.get('images'))
        ages = np.array(f.get('ages'))
        genders = np.array(f.get('gender'))
    images = images / 127.5 - 1.0
    images.astype(np.float32)
    return images, ages, genders


def categorize(ages):
    ages_copy=np.array(ages,copy=True)

    for i in range(len(ages)):
        if 0 < ages_copy[i] <= 18:
            ages_copy[i] = 0
        elif 18 < ages_copy[i] <= 29:
            ages_copy[i] = 1
        elif 29 < ages_copy[i] <= 39:
            ages_copy[i] = 2
        elif 39 < ages_copy[i] <= 49:
            ages_copy[i] = 3
        elif 49 < ages_copy[i] <= 59:
            ages_copy[i] = 4
        elif ages_copy[i] >= 60:
            ages_copy[i] = 5

    categorical = to_categorical(ages_copy)
    return categorical