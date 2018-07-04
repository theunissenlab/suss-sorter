import functools

import numpy as np
from matplotlib import cm


def make_color_map(labels):
    unique_labels = sorted(np.unique(labels))
    n_labels = len(unique_labels)
    return dict(
        (label, cm.gist_ncar(idx / n_labels))
        for idx, label in enumerate(unique_labels))


def clear_axes(*axes):
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    return axes


def get_changed_labels(new_dataset, old_dataset):
    new_labels = set(new_dataset.labels)
    old_labels = set(old_dataset.labels)
    changed_labels = set()
    for label in new_labels.union(old_labels):
        is_changed = True
        if label in old_labels and label in new_labels:
            new_node = new_dataset.nodes[new_dataset.labels == label][0]
            old_node = old_dataset.nodes[old_dataset.labels == label][0]
            if new_node == old_node:
                is_changed = False
        if is_changed:
            changed_labels.add(label)

    return changed_labels


def require_dataset(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "dataset") or not len(self.dataset):
            return None
        return func(self, *args, **kwargs)
    return wrapper
