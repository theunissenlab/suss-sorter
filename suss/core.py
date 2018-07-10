import numpy as np


class BaseDataset(object):
    """Dataset of times and raw data (i.e. spike waveforms)"""
    def __init__(self, times, data_column="datapoints", **columns):
        """
        Args
            times: Array of times
            data_column: A string used to describe the data being clustered.
                It it allows direct access to this column of the data
                through an instance attribute. Defaults to "datapoints".
            **columns: Keywords mapping column names to tuple containing
                the data to be stored and the dtype. For example,
                >>> BaseDataset(..., labels=(np.array([0, 1, 2]), ("int32")))

        Example
        >>> dataset = BaseDataset(
                times=[1, 2, 3],
                data_column="cookies",
                cookies=(np.array([[0, 1], [1, 2], [2, 3]]), ("int32", 2)),
                labels=(np.array([0, 1, 2]), ("int32"))
        >>> dataset.cookies
        [[0, 1], [1, 2], [2, 3]]
        """
        # transfrom columns from dict of {col_name: (col_data, col_dtype), ...}
        # to separate arrays of column names, data, and dtypes
        col_names, _col_data_dtype_pairs = zip(*columns.items())
        col_datas, col_dtypes = zip(*_col_data_dtype_pairs)

        times = np.array(times).flatten()
        sorter = np.argsort(times)
        _order = np.empty_like(sorter)
        _order[sorter] = np.arange(len(sorter))

        self._data = np.array(
            list(zip(times, _order, *col_datas)),
            dtype=([
                ("times", "float64"),
                ("ids", "int32")
            ] + list(zip(col_names, col_dtypes)))
        )
        self._data.sort(order="ids")
        self.data_column = data_column

        # List of Tag objects for this dataset or cluster
        # Only applies to this object! does not propogate
        # to subclusters, derived datasets, flatten(), etc...
        self._tags = set()

    @property
    def tags(self):
        if not hasattr(self, "_tags"):
            self._tags = set()
        return self._tags

    def add_tag(self, tag):
        if not hasattr(self, "_tags"):
            self._tags = set()
        self._tags.add(tag)

    def remove_tag(self, tag):
        if not hasattr(self, "_tags"):
            self._tags = set()
        return self._tags.remove(tag)

    # Set the data_column string as an accessible property
    def _get_data_column(self):
        if not self.has_children:
            return self._data[self.data_column]
        else:
            return np.array([node.centroid for node in self.nodes])

    def __getattr__(self, attr):
        """Allow access of data_column as an attribute"""
        # hack
        _get = super().__getattribute__
        if attr == _get("data_column"):
            return _get("_get_data_column")()

        return _get(attr)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        class_str = self.__class__.__name__

        if not len(self._data):
            return "Empty {}".format(class_str)

        time_str = "(time={:.3f}s)".format(self.time)

        if not self.has_children:
            contains_str = "{} {}".format(len(self._data), self.data_column)
        else:
            contains_str = "{} clusters and {} {}".format(
                len(self),
                self.count,
                self.data_column
            )

        if self.source != self:
            source_str = "\n  > derived from {}".format(self.source)
        else:
            source_str = ""

        return "{} with {} {}{}".format(
                class_str,
                contains_str,
                time_str,
                source_str)

    @property
    def has_children(self):
        return "nodes" in self._data.dtype.names

    @property
    def count(self):
        if not self.has_children:
            return len(self.ids)
        else:
            return np.sum([node.count for node in self.nodes])

    @property
    def times(self):
        return self._data["times"]

    @property
    def nodes(self):
        if not self.has_children:
            raise ValueError("Dataset is not clustered; has no 'nodes'")

        return self._data["nodes"]

    @property
    def labels(self):
        return self._data["labels"]

    @property
    def ids(self):
        return self._data["ids"]

    @property
    def centroid(self):
        """Representative datapoint is the median"""
        return np.median(getattr(self, self.data_column), axis=0)

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
        if "labels" in selected_subset.dtype.names:
            labels = selected_subset["labels"]
        else:
            labels = None
        return SubDataset(
                self,
                ids=selected_subset["ids"],
                labels=labels)

    def windows(self, dt=None, dpoints=None):
        if dpoints is not None and dt is None:
            idxs = np.arange(0, len(self.times), dpoints).astype(np.int)
            for start_idx in idxs:
                stop_idx = min(start_idx + dpoints, len(self.times))
                selector = np.arange(start_idx, stop_idx)
                yield start_idx, stop_idx, self.select(selector)
        elif dt is not None and dpoints is None:
            times = np.arange(0.0, np.max(self.times), dt)
            for t_start in times:
                t_stop = t_start + dt
                selector = np.where(
                        (self.times >= t_start) &
                        (self.times < t_stop)
                )[0]
                yield t_start, t_stop, self.select(selector)
        else:
            raise Exception("Either points or dt must be provided")

    def flatten(self, depth=None, assign_labels=True):
        if not self.has_children or (depth is not None and depth == 0):
            return self

        bottom_nodes = [
            node.flatten(None if depth is None else depth - 1)
            for node in self.nodes
        ]
        if not len(bottom_nodes):
            # FIXME (kevin): what if there are no nodes but still a source
            return self

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
            data_column=self.data_column,
            labels=unique_labels
        )


class ClusterDataset(BaseDataset):

    def __init__(self, subnodes, data_column="datapoints", labels=None):
        self.source = self
        sources = [subnode.source for subnode in subnodes]
        if sources[1:] != sources[:-1]:
            print("Warning... ClusterDataset being created with different "
                "source datasets. {}".format(subnodes))

        if labels is None:
            labels = np.zeros(len(subnodes))

        super().__init__(
            times=[subnode.time for subnode in subnodes],
            data_column=data_column,
            nodes=(subnodes, np.object),
            labels=(labels, "int32")
        )

    def select(self, selector, child=True):
        """Select items by selection array

        Args
            selector: Boolean array with length equal to number
                of clusters in dataset. Clusters corresponding
                to True will be kept
            child: Boolean flag indicating whether link to
                current dataset should be preserved. If True,
                the created SubDataset() object refers to this
                current object as its parent.  If False, returns
                a ClusterDataset with no relation to the current
                object. Defaults to True.

        Returns
            Either a SubDataset of this ClusterDataset (child=True),
            or a new ClusterDataset (child=False) whose nodes
            correspond to the selector array.
        """
        if child:
            return super().select(selector)
        else:
            return ClusterDataset(
                self.nodes[selector],
                data_column=self.data_column,
                labels=self.labels[selector]
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
            self._data["labels"] = labels
        if not all(self.ids[:-1] <= self.ids[1:]):
            self._data.sort(order="ids")
        self.data_column = self.parent.data_column

    def split(self, selector):
        """Split the SubDataset into two by selector array

        Returns
            Two SubDatasets that share the same parent.
            The first returned matches selector == True, the
            second matches selector == False

        Example:
            >>> clusters = ClusterDataset(...)
            >>> flattned = clusters.flatten()
            >>> early_cluster, late_cluster = flattened.split(
            ...     flattened.times > np.median(flattened.times)
            ... )
        """
        return self.select(selector), self.select(np.logical_not(selector))

    def time_split(self, t_start, t_stop):
        """Split node by a time range

        recursively duplicates all subnodes until the bottom level
        and reassigns their datapoints based on their time
        """
        if not self.has_children:
            selector = (self.times >= t_start) & (self.times < t_stop)
            return self.split(selector)

        within, without = zip(*[
            node.time_split(t_start, t_stop)
            for node in self.nodes
        ])
        within = [node for node in within if len(node)]
        without = [node for node in without if len(node)]
        labels = np.concatenate([
            np.zeros(len(within)),
            np.ones(len(without))
        ]).astype(np.int)

        clusters = ClusterDataset(
            np.concatenate([within, without]),
            data_column=self.data_column)
        clusters = clusters.cluster(labels)

        return (
            clusters.select(clusters.labels == 0),
            clusters.select(clusters.labels == 1)
        )

    def select(self, selector):
        """Select by index (not by id!)

        Creates a SubDataset with the same parent
        but with the selected subset of data.
        """
        selected_subset = self._data[selector]
        return SubDataset(
                self.parent,
                ids=selected_subset["ids"],
                labels=selected_subset["labels"],
                source_dataset=self.source
        )


class SpikeDataset(BaseDataset):

    def __init__(self, times, waveforms, sample_rate=None, labels=None):
        self.source = self
        self.sample_rate = sample_rate
        if labels is None:
            labels = np.zeros(len(times))

        super().__init__(
            times=times,
            data_column="waveforms",
            waveforms=(waveforms, ("float64", waveforms.shape[1])),
            labels=(labels, "int32")
        )
