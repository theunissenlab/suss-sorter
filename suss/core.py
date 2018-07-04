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

        times = np.array(times).flatten()
        sorter = np.argsort(times)
        _order = np.empty_like(sorter)
        _order[sorter] = np.arange(len(sorter))

        self._data = np.array(
            list(zip(times, _order, *col_datas)),
            dtype=([
                ("time", "float64"),
                ("id", "int32")
            ] + list(zip(col_names, col_dtypes)))
        )
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
                selector = np.where((self.times >= t_start) & (self.times < t_stop))[0]
                yield t_start, t_stop, self.select(selector)
        else:
            raise Exception("Either points or dt must be provided")


    def flatten(self, depth=None, assign_labels=True):
        if self.is_waveform or (depth is not None and depth == 0):
            return self

        bottom_nodes = [
            node.flatten(None if depth is None else depth - 1)
            for node in self.nodes
        ]
        if not len(bottom_nodes):
            return self  # FIXME (kevin): this is probably not the right way to handle this

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

    def __init__(self, times, waveforms, sample_rate=None, labels=None):
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

    def select(self, selector, child=True):
        """Select items by selection array

        Args
            selector: Boolean array with length equal to number of
                clusters in dataset. Clusters corresponding to True will
                be kept
            child: Boolean flag indicating whether link to current dataset
                should be preserved. If True, the created SubDataset()
                object refers to this current object as its parent.
                If False, returns a ClusterDataset with no relation to the
                current object. Defaults to True.

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
                labels=self.labels[selector]
            )

    def delete_node(self, node=None, idx=None, label=None):
        if np.sum([node is not None, idx is not None, label is not None]) != 1:
            raise ValueError("Only one of {node, idx, label} can be provided")
        if label is not None:
            match = np.where(self.labels == label)[0]
        elif node is not None:
            match = np.where(self.nodes == node)[0]
        elif idx is not None:
            match = [idx]

        if len(match) > 1:
            raise ValueError("More than one node matched label {}".format(label))
        elif len(match) == 0:
            raise ValueError("No node matched label {}".format(label))
        else:
            idx = match[0]

        selector = np.eye(len(self))[idx].astype(np.bool)

        return self.select(np.logical_not(selector), child=False)

    def add_nodes(self, *nodes):
        if not len(self.labels):
            start_at = 0
        else:
            start_at = np.max(self.labels)
            
        new_labels = np.arange(
                start_at + 1,
                start_at + len(nodes) + 1
        )
        return ClusterDataset(
            np.concatenate([self.nodes, nodes]),
            labels=np.concatenate([self.labels, new_labels])
        )

    def uncluster_node(self, node=None, idx=None, label=None):
        if np.sum([node is not None, idx is not None, label is not None]) != 1:
            raise ValueError("Only one of {node, idx, label} can be provided")
        if label is not None:
            match = np.where(self.labels == label)[0]
        elif node is not None:
            match = np.where(self.nodes == node)[0]
        elif idx is not None:
            match = [idx]

        if len(match) > 1:
            raise ValueError("More than one node matched label {}".format(label))
        elif len(match) == 0:
            raise ValueError("No node matched label {}".format(label))
        else:
            idx = match[0]

        node = self.nodes[idx]
        new_dataset = self.delete_node(idx=idx)
        new_dataset = new_dataset.add_nodes(*node.nodes)

        return new_dataset

    def split_node(
            self,
            t_start,
            t_stop,
            node=None,
            idx=None,
            label=None,
            keep_both=True):

        if np.sum([node is not None, idx is not None, label is not None]) != 1:
            raise ValueError("Only one of {node, idx, label} can be provided")
        if label is not None:
            match = np.where(self.labels == label)[0]
        elif node is not None:
            match = np.where(self.nodes == node)[0]
        elif idx is not None:
            match = [idx]

        if len(match) > 1:
            raise ValueError("More than one node matched label {}".format(label))
        elif len(match) == 0:
            raise ValueError("No node matched label {}".format(label))
        else:
            idx = match[0]

        selector = np.eye(len(self))[idx].astype(np.bool)
        in_range, out_range = self.nodes[idx].time_split(t_start, t_stop)
        new_dataset = self.select(np.logical_not(selector), child=False)

        if len(in_range) > 0:
            new_dataset = new_dataset.add_nodes(in_range)
        if keep_both and len(out_range) > 0:
            new_dataset = new_dataset.add_nodes(out_range)

        return new_dataset

    def merge_nodes(self, labels=None, idxs=None, nodes=None):
        if np.sum([nodes is not None, idxs is not None, labels is not None]) != 1:
            raise ValueError("Only one of {nodes, idxs, labels} can be provided")
        if labels is not None:
            match = np.where(np.isin(self.labels, list(labels)))[0]
        elif node is not None:
            match = np.where(np.isin(self.nodes, list(nodes)))[0]
        elif idx is not None:
            match = idxs

        selector = np.zeros(len(self)).astype(np.bool)
        selector[match] = True

        nodes = self.select(selector).nodes

        merged = nodes[0].merge(*nodes[1:])
        new_dataset = self.select(np.logical_not(selector), child=False)
        new_dataset = new_dataset.add_nodes(merged)

        return new_dataset


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

    @property
    def complement(self):
        return self.source.complement(self)

    def merge(self, *nodes):
        """Merge this node with one or more other nodes"""
        all_nodes = [self] + list(nodes)
        return SubDataset(
            self.parent,
            ids=np.concatenate([node.ids for node in all_nodes]),
            source_dataset=self.source
        )

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
        if self.is_waveform:
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

        clusters = ClusterDataset(np.concatenate([within, without]))
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
                ids=selected_subset["id"],
                labels=selected_subset["label"],
                source_dataset=self.source
        )
