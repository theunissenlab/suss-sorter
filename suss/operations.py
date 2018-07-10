"""Operations for modifying the clusters in a dataset
"""
import numpy as np
from sklearn.cluster import KMeans

from suss.core import ClusterDataset, SubDataset
from suss.sort import pca_time, cleanup_clusters, tsne_time


def force_single_kwarg(**kwargs):
    keys = list(kwargs.keys())
    new_kwargs = dict((k, v) for k, v in kwargs.items() if v is not None)
    if len(new_kwargs) != 1:
        raise ValueError(
            "Only one of {} can be provided, "
            "received {}".format(keys, list(new_kwargs.keys())))

    return new_kwargs


def match_several(dataset, labels=None, nodes=None, idxs=None):
    force_single_kwarg(labels=labels, idxs=idxs, nodes=nodes)
    if labels is not None:
        match = np.where(np.isin(dataset.labels, list(labels)))[0]
    elif nodes is not None:
        match = np.where(np.isin(dataset.nodes, list(nodes)))[0]
    elif idxs is not None:
        match = idxs

    selector = np.zeros(len(dataset)).astype(np.bool)
    selector[match] = True
    return selector


def match_one(dataset, label=None, node=None, idx=None):
    kw = force_single_kwarg(label=label, idx=idx, node=node)
    if label is not None:
        match = np.where(dataset.labels == label)[0]
    elif node is not None:
        match = np.where(dataset.nodes == node)[0]
    elif idx is not None:
        match = [idx]

    if len(match) > 1:
        raise ValueError("More than one node matched {}".format(kw))
    elif len(match) == 0:
        raise ValueError("No node matched {}".format(kw))
    else:
        idx = match[0]

    selector = np.eye(len(dataset))[idx].astype(np.bool)
    return selector


def _merge(*nodes):
    """Merge multiple SubDatasets into 
    """
    if not all([isinstance(node, SubDataset) for node in nodes]):
        raise ValueError(
            "All arguments to _merge_nodes must be "
            "SubDataset instances")

    parents = [node.parent for node in nodes]
    sources = [node.source for node in nodes]

    if len(np.unique(parents)) > 1:
        raise ValueError(
            "Cannot combine nodes with different "
            "parents: {}".format(np.unique(parents)))

    if len(np.unique(sources)) > 1:
        raise ValueError(
            "Cannot combine nodes with different "
            "sources: {}".format(np.unique(sources)))

    return SubDataset(
        parents[0],
        ids=np.concatenate([node.ids for node in nodes]),
        source_dataset=sources[0]
    )


def add_nodes(dataset, *nodes):
    """Add new nodes to a dataset"""
    if not len(dataset.labels):
        start_at = 0
    else:
        start_at = np.max(dataset.labels)

    new_labels = np.arange(
        start_at + 1,
        start_at + len(nodes) + 1
    )
    return ClusterDataset(
        np.concatenate([dataset.nodes, nodes]),
        data_column=dataset.data_column,
        labels=np.concatenate([dataset.labels, new_labels])
    )


def delete_nodes(dataset, nodes=None, idxs=None, labels=None):
    selector = match_several(dataset, nodes=nodes, idxs=idxs, labels=labels)
    return dataset.select(np.logical_not(selector), child=False)


def delete_node(dataset, node=None, idx=None, label=None):
    selector = match_one(dataset, node=node, idx=idx, label=label)
    return dataset.select(np.logical_not(selector), child=False)


def merge_nodes(dataset, nodes=None, idxs=None, labels=None):
    """Merge nodes of a dataset together.
    """
    selector = match_several(dataset, labels=labels, idxs=idxs, nodes=nodes)

    new_dataset = dataset.select(np.logical_not(selector), child=False)
    nodes_to_combine  = dataset.select(selector).nodes
    return add_nodes(new_dataset, _merge(*nodes_to_combine))


def recluster_node(dataset, node=None, idx=None, label=None):
    selector = match_one(dataset, label=label, idx=idx, node=node)

    # Get the node you want to recluster and flatten it
    selected_data = dataset.select(selector).flatten(1)
    cluster_on = pca_time(selected_data, pcs=6, t_scale=30 * 60 * 60.0)

    # Reassociate that data with new labels
    kmeans = KMeans(n_clusters=4).fit(cluster_on)
    labels = kmeans.predict(cluster_on)
    labels = cleanup_clusters(cluster_on, labels, n_neighbors=3)
    reclustered = selected_data.cluster(labels)

    new_dataset = dataset.select(np.logical_not(selector), child=False)
    return add_nodes(new_dataset, *reclustered.nodes)

    

'''
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
            raise ValueError("More than one node matched "
                    "label {}".format(label))
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
'''
