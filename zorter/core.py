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
    def has_children(self):
        return "waveform" in self._data.dtype.names

    @property
    def times(self):
        return self._data["time"]

    @property
    def ids(self):
        return self._data["id"]

    def select(self, selector):
        selected_subset = self._data[selector]
        return DataNode(self, ids=selected_subset["id"])

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

    @property
    def waveforms(self):
        return self._data["waveform"]

    @property
    def labels(self):
        return self._data["label"]


class ClusterDataset(BaseDataset):

    def __init__(self, child_nodes, labels=None):
        if labels is None:
            labels = np.zeros(len(child_nodes))

        super().__init__(
            times=[node.time for node in child_nodes],
            node=(child_nodes, np.object),
            label=(labels, "int32")
        )

    @property
    def waveforms(self):
        return np.array([node.waveform for node in self._data["node"]])

    @property
    def labels(self):
        return self._data["label"]

    @property
    def nodes(self):
        return self._data["node"]


class DataNode(object):
    """Represents a subset of data in a dataset

    This can act as its own data point in a hierarchical clustering algorithm
    with a representative waveform shape (the median / mean of its child
    nodes), and representative time (the median time of its child nodes)
    """

    def __init__(self, dataset, ids):
        self.dataset = dataset
        self.ids = ids

    @property
    def waveforms(self):
        return self.dataset.waveforms[self.ids]

    @property
    def times(self):
        return self.dataset.times[self.ids]

    @property
    def waveform(self):
        return np.median(self.waveforms, axis=0)

    @property
    def time(self):
        return np.median(self.times)

    @property
    def children(self):
        return self.dataset.nodes[self.ids]

    def merge(self, *nodes):
        return self.dataset.merge(*([node] + nodes))

    @property
    def complement(self):
        return self.dataset.complement(self)

    def split(self):
        return self.dataset.split(self)

    def flatten(self, depth=None, assign_labels=True):
        if not self.dataset.has_children or (depth is not None and depth == 0):
            return self

        flattened_children = [
            child.flatten(None if depth is None else depth - 1)
            for child in self.children
        ]

        if np.unique([child.dataset.has_children for child in flattened_children]).size != 1:
            raise Exception("DataNode tree branches have different depths... "
                    "something must be terribly wrong.")

        next_dataset = flattened_children[0].dataset

        if assign_labels:
            labels = np.concatenate([
                idx * np.ones(len(child.times))
                for idx, child in enumerate(flattened_children)
            ])

        return DataNode(
            next_dataset,
            np.concatenate([child.ids for child in flattened_children])
        )
