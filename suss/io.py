import pickle
import numpy as np
import scipy
import os


def read_numpy(filename):
    return np.load(filename, allow_pickle=True)[()]


def save_numpy(filename, data):
    np.save(filename, data)


def read_pickle(filename):
    with open(filename, "rb") as open_file:
        return pickle.load(open_file)


def save_pickle(filename, data):
    with open(filename, "wb+") as open_file:
        pickle.dump(data, open_file)


def read_mat(filename):
    raise IOError(".mat compatibility not supported yet")
