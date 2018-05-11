import numpy as np


class BaseDataset(object):
    """Spike dataset of times and waveforms"""
    def __init__(self, times, **columns):
        """
        Args
            times: Array of times
            **columns: Keywords mapping column names to tuple containing
                the data to be stored and the dtype. For example,
                >>> Dataset(times, labels=(np.array([0, 1, 2]), ("int32")))
        """
        # transfrom columns from dict of {col_name: (col_data, col_dtype), ...}
        # to separate arrays of column names, data, and dtypes
        col_names, _col_data_dtype_pairs = zip(*columns.items())
        col_datas, col_dtypes = zip(*_col_data_dtype_pairs)

        _order = np.argsort(times)
        self._data = np.array(
            list(zip(times, _order, *col_datas)),
            dtype=([
                ("time", "float64"),
                ("id", "int32")
            ] + list(zip(col_names, col_dtypes)))
        )

        # id should be a column that is always the same as the
        # normal index by which the array is accessed.
        # Ensure this by sorting by this column
        self._data.sort(order="id")

    @property
    def is_waveform(self):
        return "waveform" in self._data.dtype.names

    @property
    def times(self):
        return self._data["time"]

    @property
    def waveforms(self):
        if self.is_waveform:
            return self._data["waveform"]
        else:
            return np.array([node.waveform for node in self.nodes])

    @property
    def nodes(self):
        if self.is_waveform:
            raise ValueError("Dataset containing waveforms has no 'nodes'")

        return self._data["node"]

    @property
    def labels(self):
        return self._data["label"]

    @property
    def ids(self):
        return self._data["id"]

    def select(self, selector):
        selected_subset = self._data[selector]
        return SubDataset(self, ids=selected_subset["id"])

    def merge(self, *nodes):
        ids = np.concatenate([node.ids for node in nodes])
        return self.select(ids)

    def complement(self, node):
        return self.select(np.delete(self.ids, node.ids))

    def split(self, *nodes):
        """Divide dataset into datanodes containing given nodes and node excluding
        """
        combined = self.merge(nodes)
        return combined, self.complement(combined)

    def windows(self, dt):
        for t_start in np.arange(0.0, np.max(self.times), dt):
            t_stop = t_start + dt
            selector = np.where((self.times >= t_start) & (self.times < t_stop))[0]
            yield t_start, t_stop, self.select(selector)

    def flatten(self, depth=None, assign_labels=True):
        if self.is_waveform or (depth is not None and depth == 0):
            return self

        bottom_nodes = [
            node.flatten(None if depth is None else depth - 1)
            for node in self.nodes
        ]
        bottom_dataset = bottom_nodes[0].parent
        bottom_ids = [node.ids for node in bottom_nodes]

        if assign_labels:
            labels = np.concatenate([
                node_idx * np.ones(len(ids))
                for node_idx, ids in enumerate(bottom_ids)
            ])
        else:
            labels = None

        return SubDataset(bottom_dataset, np.concatenate(bottom_ids), labels=labels)

    def cluster(self, cluster_labels):
        return ClusterDataset([
            self.select(cluster_labels == label)
            for label in np.unique(cluster_labels)
        ])


class SpikeDataset(BaseDataset):

    def __init__(self, times, waveforms, sample_rate, labels=None):
        self.sample_rate = sample_rate
        if labels is None:
            labels = np.zeros(len(times))

        super().__init__(
            times=times,
            waveform=(waveforms, ("float64", waveforms.shape[1])),
            label=(labels, "int32")
        )


class ClusterDataset(BaseDataset):

    def __init__(self, nodes, labels=None):
        if not all([hasattr(node, "parent") for node in nodes]):
            raise ValueError("Cannot generate a ClusterDataset from nodes without "
                    "parents (must be SubDataset objects)")

        if labels is None:
            labels = np.zeros(len(nodes))

        super().__init__(
            times=[node.time for node in nodes],
            node=(nodes, np.object),
            label=(labels, "int32")
        )


class SubDataset(BaseDataset):
    """Represents a subset of data in a dataset

    This can act as its own data point in a hierarchical clustering algorithm
    with a representative waveform shape (the median / mean of its child
    nodes), and representative time (the median time of its child nodes)
    """
    def __init__(self, parent_dataset, ids, labels=None):
        self.parent = parent_dataset

        # Copy the selected subset of the parent's data.
        # If you are seeing unexpected behavior from this, perhaps
        # looking here may be useful:
        # http://scipy-cookbook.readthedocs.io/items/ViewsVsCopies.html
        self._data = self.parent._data[ids]
        if labels is not None:
            self._data["label"] = labels 
        self._data.sort(order="id")

    @property
    def waveform(self):
        return np.median(self.waveforms, axis=0)

    @property
    def time(self):
        return np.median(self.times)

    def merge(self, *nodes):
        return self.parent.merge(*([self] + nodes))

    @property
    def complement(self):
        return self.parent.complement(self)

    def split(self):
        return self.parent.split(self)

    def select(self, *args, **kwargs):
        raise NotImplementedError("Cannot select by ids from SubDataset. "
                "Did you want to select from its parent?")
