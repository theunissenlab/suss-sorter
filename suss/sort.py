import time

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import KNeighborsClassifier

try:
    from MulticoreTSNE import MulticoreTSNE as TSNE
except ImportError:
    from sklearn.manifold import TSNE

from .core import ClusterDataset, SpikeDataset, SubDataset


def cluster(dataset, n_components=2, mode= "kmeans", transform=None):
    """Split node into several by clustering

    Create several new nodes from parent by clustering. This is basically
    an abstraction of KMeans and BayesianGaussianMixture model syntax.

    Args:
        node: A instance of a core.BaseDataset whose waveforms will be clustered
        n_components: An integer number representing the (maximum) number
            of clusters to generate
        mode (default: "kmeans"): the clustering algorithm to apply. Can be
            'kmeans' or 'gmm'
        transform (optional): function that maps waveforms to a new feature space

    Returns:
        A list of core.SubDataset objects whose data represents the result
        of clustering
    """
    if mode not in ("kmeans", "gmm", "spectral"):
        raise ValueError("mode must be either 'kmeans' or 'gmm' or 'spectral'")

    n_components = min(n_components, len(dataset.waveforms))

    if not len(dataset.waveforms):
        return dataset.cluster([]).nodes

    data = transform(dataset.waveforms) if transform is not None else dataset.waveforms

    if mode == "tsne-dbscan":
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        data = TSNE(n_components=2, perplexity=5.0, n_iter=2000).fit_transform(data)
        labels = clusterer.fit_predict(data)
        # replace -1 labels with unique labels (so they can get pruned)
        bad_labels = np.where(labels == -1)[0]
        _fill_in = np.arange(np.max(labels) + 1, np.max(labels) + 1 + len(bad_labels))
        labels[bad_labels] = _fill_in
    elif mode == "kmeans":
        clusterer = KMeans(n_clusters=n_components)
        clusterer.fit(data)
        labels = clusterer.predict(data)
    elif mode == "gmm":
        clusterer = BayesianGaussianMixture(n_components=n_components)
        clusterer.fit(data)
        labels = clusterer.predict(data)
    elif mode == "spectral":
        clusterer = SpectralClustering(n_clusters=n_components)
        labels = clusterer.fit_predict(data)
    
    return dataset.cluster(labels).nodes


def cluster_step(dataset, dt=None, dpoints=None, n_components=2, mode="kmeans", transform=None):
    """Implement a first step of the hierarchical clustering algorithm

    From a single core.ClusterDataset or core.SpikeDataset, apply clustering over
    time windows of duration dt, and create a new core.ClusterDataset whose nodes
    represent data clustered in this process.
    
    Args:
        node: An instance of core.BaseDataset whose waveforms will be clustered
        dt: duration (in seconds) of time window to cluster within
        n_clusters: An integer number representing the (maximum) number
            of clusters to generate
        mode (default: "kmeans"): the clustering algorithm to apply. Can be
            'kmeans' or 'gmm'
        transform (optional): function that maps waveforms to a new feature space

    Returns:
        A core.ClusterDataset object with one child for each cluster at
        each timestep.
    """
    _denoised_nodes = []
    _fn_start = time.time()

    # new clustering over time thing
    for t_start, _, window in dataset.windows(dt=dt, dpoints=dpoints):
        print("Clustering t={:.2f}.min. {:.1f}s elapsed.".format(
            t_start / 60.0, time.time() - _fn_start), end="\r")
        _denoised_nodes.append(
            cluster(
                window,
                n_components=n_components,
                mode=mode,
                transform=transform
            )
        )

    if dt:
        print("Completed clustering of {:.2f} min in {:.1f}s.".format(
            (t_start + dt) / 60.0, time.time() - _fn_start))
    else:
        print("Completed clustering of {} points in {:.1f}s.".format(
            (t_start + dpoints), time.time() - _fn_start))

    return ClusterDataset(np.concatenate(_denoised_nodes))


def cluster_step(dataset, dpoints=None, n_components=2, mode="kmeans", transform=None):
    """Implement a first step of the hierarchical clustering algorithm

    From a single core.ClusterDataset or core.SpikeDataset, apply clustering over
    time windows of duration dt, and create a new core.ClusterDataset whose nodes
    represent data clustered in this process.
    
    Args:
        node: An instance of core.BaseDataset whose waveforms will be clustered
        dpoints: Number of points to take in each cluster step
        n_clusters: An integer number representing the (maximum) number
            of clusters to generate
        mode (default: "kmeans"): the clustering algorithm to apply. Can be
            'kmeans' or 'gmm'
        transform (optional): function that maps waveforms to a new feature space

    Returns:
        A core.ClusterDataset object with one child for each cluster at
        each timestep.
    """
    _denoised_nodes = []
    _fn_start = time.time()

    _new_labels = -1 * np.ones(len(dataset)).astype(np.int)
    while -1 in _new_labels:
        next_window = np.where(_new_labels == -1)[0][:dpoints]

        if len(next_window) < n_components:
            break

        if len(next_window) < dpoints and len(next_window) == len_last_window:
            break

        len_last_window = len(next_window)

        window_data = dataset.waveforms[next_window]
        clusterer = KMeans(n_clusters=n_components)
        clusterer.fit(window_data)
        labels = clusterer.predict(window_data)

        for label, count in zip(*np.unique(labels, return_counts=True)):
            if count < 10:
                labels[labels == label] = -1
            else:
                labels[labels == label] += np.max(_new_labels) + 1

        _new_labels[next_window] = labels
        print("Completed {}/{} in {:.1f}s.".format(
            np.max(next_window), len(dataset), time.time() - _fn_start), end="\r")

    print("Completed clustering in {:.1f}s".format(time.time() - _fn_start))
    return dataset.cluster(_new_labels)


def reassign_unassigned(waveforms, labels):
    neigh = KNeighborsClassifier(n_neighbors=1)
    if len(np.where(labels != -1)[0]) == 0:
        return labels

    neigh.fit(waveforms[labels != -1], labels[labels != -1])
    labels[np.where(labels == -1)] = neigh.predict(waveforms[np.where(labels == -1)])
    return labels


def denoise_step(dataset, current_node, min_waveforms, dt=None, dpoints=None, n_components=None, mode=None):
    """Perform clustering and then reassign the cluster centroid values to the original datapoints
    """
    denoised_node = cluster_step(
        current_node.flatten(),
        dpoints=dpoints,
        n_components=n_components,
        mode=mode
    )

    flat = denoised_node.flatten(assign_labels=True)
    centroids = dict(
        (
            label,
            np.median(flat.waveforms[flat.labels == label], axis=0)
        )
        for label in np.unique(flat.labels)
    )

    if len(flat.ids):
        dataset.waveforms[flat.ids] = [centroids[label] for label in flat.labels]

    return denoised_node


def denoising_sort(times, waveforms):
    spike_dataset = SpikeDataset(times=times, waveforms=waveforms)

    original_waveforms = spike_dataset.waveforms.copy()

    steps = [
        dict(min_waveforms=2, dpoints=1000, n_components=32, mode="kmeans"),
        dict(min_waveforms=15, dpoints=2000, n_components=16, mode="kmeans"),
    ]


    dataset = spike_dataset
    denoised_node = dataset
    try:
        for step_kwargs in steps:
            denoised_node = denoise_step(dataset, denoised_node, **step_kwargs)
    except:
        raise
    finally:
        spike_dataset.waveforms[:] = original_waveforms

    return denoised_node


def sort(times, waveforms):
    denoised = denoising_sort(times, waveforms)
    tsned = TSNE(n_components=2).fit_transform(
        PCA(n_components=20).fit_transform(denoised.waveforms)
    )

    # Now perform another TSNE, incorporating time
    time_arr = denoised.times / (60.0 * 60.0)  # divide by 1 hr
    time_arr = time_arr - np.mean(time_arr)
    wf_arr = scipy.stats.zscore(tsned, axis=0)

    tsned_with_time = TSNE(n_components=2).fit_transform(
        np.hstack([wf_arr, time_arr[:, None]])
    )

    # Try different clusterings until the number of clusters falls
    # within our desired range
    # TODO (kevin): improve this selection
    for n in range(20, 8, -2):
        hdb = hdbscan.HDBSCAN(min_cluster_size=n)
        final_labels = hdb.fit_predict(tsned_with_time)
        if 20 <= len(np.unique(final_labels)) <= 40:
            print("Selected min_cluster_size={}".format(n))
            break

    final_labels = reassign_unassigned(tsned_with_time, final_labels)

    return denoised.cluster(final_labels)
