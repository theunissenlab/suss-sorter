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
        if not all(self.ids[:-1] <= self.ids[1:]):
            self._data.sort(order="id")
        self._data.sort(order="id")

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        class_str = self.__class__.__name__

        if not len(self._data):
            return "Empty {}".format(class_str)

        time_str = "(time={:.3f}s)".format(self.time)

        if self.is_waveform:
            contains_str = "{} waveforms".format(len(self._data))
        else:
            contains_str = "{} clusters and {} waveforms".format(
                len(self),
                self.waveform_count
            )

        if self.source != self:
            source_str = "\n  > derived from {}".format(self.source)
        else:
            source_str = ""

        return "{} with {} {}{}".format(class_str,
                contains_str,
                time_str,
                source_str)

    @property
    def is_waveform(self):
        return "waveform" in self._data.dtype.names

    @property
    def waveform_count(self):
        if self.is_waveform:
            return len(self.ids)
        else:
            return np.sum([node.waveform_count for node in self.nodes])

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

    @property
    def waveform(self):
        """Representative waveform is the median waveform"""
        return np.median(self.waveforms, axis=0)

    @property
    def time(self):
        """Representative time is the median time"""
        return np.median(self.times)

    def __lt__(self, other):
        """Order objects by their median time

        Implemented so that arrays containing these objects can be sorted
        """
        return self.time < other.time

    def select(self, selector):
        """Select by index (not by id!)"""
        selected_subset = self._data[selector]
        return SubDataset(self,
                ids=selected_subset["id"],
                labels=selected_subset["label"])

    def merge(self, *nodes):
        ids = np.concatenate([node.ids for node in nodes])
        return self.select(ids)

    def complement(self, node):
        return self.select(np.delete(self.ids, node.ids))

    def split(self, *nodes):
        """Divide dataset into datanodes containing given nodes and node excluding
        """
        combined = self.merge(*nodes)
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
        bottom_dataset = bottom_nodes[0].source
        bottom_ids = [node.ids for node in bottom_nodes]

        if assign_labels:
            if len(np.unique(self.labels)) == len(bottom_nodes):
                labels = np.concatenate([
                    current_label * np.ones(len(ids))
                    for current_label, ids in zip(self.labels, bottom_ids)
                ])
            else:
                labels = np.concatenate([
                    node_idx * np.ones(len(ids))
                    for node_idx, ids in enumerate(bottom_ids)
                ])
        else:
            labels = None

        return SubDataset(
                bottom_dataset,
                np.concatenate(bottom_ids),
                labels=labels)

    def cluster(self, cluster_labels):
        unique_labels = sorted(np.unique(cluster_labels))
        return ClusterDataset(
            [
                self.select(cluster_labels == label)
                for label in unique_labels
            ],
            labels=unique_labels
        )


class SpikeDataset(BaseDataset):

    def __init__(self, times, waveforms, sample_rate, labels=None):
        self.source = self
        self.sample_rate = sample_rate
        if labels is None:
            labels = np.zeros(len(times))

        super().__init__(
            times=times,
            waveform=(waveforms, ("float64", waveforms.shape[1])),
            label=(labels, "int32")
        )


class ClusterDataset(BaseDataset):

    def __init__(self, subnodes, labels=None):
        self.source = self
        sources = [subnode.source for subnode in subnodes]
        if sources[1:] != sources[:-1]:
            print("Warning... ClusterDataset being created with different "
                    "source datasets. {}".format(subnodes))

        if labels is None:
            labels = np.zeros(len(subnodes))

        super().__init__(
            times=[subnode.time for subnode in subnodes],
            node=(subnodes, np.object),
            label=(labels, "int32")
        )


class SubDataset(BaseDataset):
    """Represents a subset of data in a dataset

    This can act as its own data point in a hierarchical clustering algorithm
    with a representative waveform shape (the median / mean of its child
    nodes), and representative time (the median time of its child nodes)
    """
    def __init__(self, parent_dataset, ids, source_dataset=None, labels=None):
        self.parent = parent_dataset
        self.source = source_dataset or parent_dataset 

        # Copy the selected subset of the parent's data.
        # If you are seeing unexpected behavior from this, perhaps
        # looking here may be useful:
        # http://scipy-cookbook.readthedocs.io/items/ViewsVsCopies.html
        self._data = self.source._data[ids]
        if labels is not None:
            self._data["label"] = labels
        if not all(self.ids[:-1] <= self.ids[1:]):
            self._data.sort(order="id")

    def merge(self, *nodes):
        return self.source.merge(*((self,) + nodes))

    @property
    def complement(self):
        return self.source.complement(self)

    def split(self):
        return self.source.split(self)

    def select(self, selector):
        """Select by index (not by id!)"""
        selected_subset = self._data[selector]
        return SubDataset(
                self.parent,
                ids=selected_subset["id"],
                labels=selected_subset["label"],
                source_dataset=self.source
        )
