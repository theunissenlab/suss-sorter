import numpy as np
from matplotlib import cm


def make_color_map(labels):
    unique_labels = sorted(np.unique(labels))
    n_labels = len(unique_labels)
    return dict((label, cm.gist_ncar(idx / n_labels))
            for idx, label in enumerate(unique_labels))


def clear_axes(*axes):
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    return axes
