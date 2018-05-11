import pickle
import numpy as np
import scipy

from .sparse import Dictionary


def read_numpy(filename):
    return np.load(filename)[()]


def save_numpy(filename, data):
    return np.save(filename, data)


def read_pickle(filename):
    with open(filename, "rb") as open_file:
        return pickle.load(open_file)


def save_pickle(filename, data):
    with open(filename, "wb") as open_file:
        pickle.dump(data, open_file)


def read_mat(filename):
    raise IOError(".mat compatibility not supported yet")


def load_dictionary(filename):
    keys = read_numpy(filename)
    assert "components" in keys
    assert "sample_rate" in keys
    assert "center_bin" in keys

    return Dictionary(**keys)


def save_dictionary(filename, dictionary):
    to_save = {
            "components": dictionary.components,
            "sample_rate": dictionary.sample_rate,
            "center_bin": dictionary.center_bin
    }
    np.save(filename, to_save)
