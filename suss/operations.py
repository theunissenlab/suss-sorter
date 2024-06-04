"""Operations for modifying the clusters in a dataset
"""
import networkx as nx
import numpy as np
import scipy

import umap
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, kneighbors_graph
from sklearn.neighbors import LocalOutlierFactor

from suss.core import ClusterDataset, SubDataset
from suss.sort import pca_time, cleanup_clusters, tsne_time


def remove_outliers(mknn, n_neighbors=10, edges=1):    
    degs = [n for n, d in mknn.degree() if d < edges + 1]
    while len(degs):
        mknn.remove_nodes_from(degs)
        degs = [n for n, d in mknn.degree() if d < edges + 1]

    return mknn


def label_outliers(X, p=0.1, n_neighbors=10):
    """GMM Version"""
    if len(X) <= 4:
        return np.ones(len(X))

    pcs = PCA(n_components=2 if X.shape[1] >= 2 else 1).fit_transform(X)
    scores = np.max(np.abs(scipy.stats.zscore(pcs, axis=0)), axis=1)
    return scores < 2


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

    selector = np.zeros(len(dataset)).astype(bool)
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

    selector = np.eye(len(dataset))[idx].astype(bool)
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


def recluster_node(dataset, node=None, idx=None, label=None, n_clusters=4):
    selector = match_one(dataset, label=label, idx=idx, node=node)

    # Get the node you want to recluster and flatten it
    selected_data = dataset.select(selector).flatten(1)
    if len(selected_data) >= 100:
        cluster_on = tsne_time(selected_data, pcs=6, t_scale=2 * 60 * 60.0)
    else:
        cluster_on = PCA(
            n_components=min(6, len(selected_data))
        ).fit_transform(selected_data.waveforms)

    n_clusters = min(n_clusters, len(cluster_on))

    weight = np.array([node.count for node in selected_data.nodes])

    # kmeans = KMeans(n_clusters=n_clusters).fit(cluster_on, sample_weight=weight)
    # labels = kmeans.predict(cluster_on, sample_weight=weight)
    if len(cluster_on) < 2:
        labels = np.arange(len(cluster_on))
    else:
        gmm = BayesianGaussianMixture(n_components=n_clusters).fit(cluster_on)
        labels = gmm.predict(cluster_on)
    reclustered = selected_data.cluster(labels)

    new_dataset = dataset.select(np.logical_not(selector), child=False)
    return add_nodes(new_dataset, *reclustered.nodes)


def cleanup_node(dataset, node=None, idx=None, label=None, n_clusters=3):
    selector = match_one(dataset, label=label, idx=idx, node=node)

    # Get the node you want to recluster and flatten it
    selected_data = dataset.select(selector).flatten(1)
    # proj = umap.UMAP(n_components=6).fit_transform(selected_data.waveforms)
    if len(selected_data.waveforms) > 3:
        proj = PCA(n_components=3).fit_transform(selected_data.waveforms)
        outliers = label_outliers(proj, p=0.01)
    else:
        outliers = np.arange(len(selected_data.waveforms))

    new_dataset = dataset.select(np.logical_not(selector), child=False)
    return add_nodes(
            new_dataset,
            *selected_data.cluster(outliers).nodes
    )


def recluster_node_in_time(dataset, node=None, idx=None, label=None, n_clusters=3):
    selector = match_one(dataset, label=label, idx=idx, node=node)

    # Get the node you want to recluster and flatten it
    all_selected_data = dataset.select(selector).flatten(1)
    # proj = PCA(n_components=6).fit_transform(all_selected_data.waveforms)
    outliers = label_outliers(all_selected_data.times[:, None], p=0.01)

    selected_data = all_selected_data.select(outliers == 1)
    outlier_data = all_selected_data.select(outliers == 0)

    # selected_data = dataset.select(selector).flatten(1)
    cluster_on = selected_data.times[:, None]
    weight = np.array([node.count for node in selected_data.nodes])

    n_clusters = min(n_clusters, len(cluster_on))

    new_dataset = dataset.select(np.logical_not(selector), child=False)

    if len(cluster_on):
        kmeans = KMeans(n_clusters=n_clusters).fit(cluster_on, sample_weight=weight)
        labels = kmeans.predict(cluster_on, sample_weight=weight)
        reclustered = selected_data.cluster(labels)
    else:
        reclustered = selected_data.cluster(np.array([]))

    if len(outlier_data) and len(reclustered):
        return add_nodes(new_dataset, outlier_data, *reclustered.nodes)
    else:
        return add_nodes(new_dataset, *reclustered.nodes)


def cleanup_cluster_assignments(dataset, n_neighbors=3):
    flat = dataset.flatten(1)
    projection = pca_time(flat, pcs=6, t_scale=1 * 60 * 60.0)
    new_labels = cleanup_clusters(projection, flat.labels, n_neighbors=n_neighbors)
    return flat.cluster(new_labels)

