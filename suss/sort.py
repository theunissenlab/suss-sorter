import time

import hdbscan
import numpy as np
import scipy.stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

try:
    from MulticoreTSNE import MulticoreTSNE as TSNE
except ImportError:
    from sklearn.manifold import TSNE

from .core import SpikeDataset


def cluster_step(
        dataset,
        dpoints=None,
        n_components=2,
        mode="kmeans",
        transform=None):
    """Implement a first step of the hierarchical clustering algorithm

    From a single core.ClusterDataset or core.SpikeDataset, apply clustering
    over time windows of duration dt, and create a new core.ClusterDataset
    whose nodes represent data clustered in this process.

    Args:
        node: An instance of core.BaseDataset whose waveforms will be clustered
        dpoints: Number of points to take in each cluster step
        n_clusters: An integer number representing the (maximum) number
            of clusters to generate
        mode (default: "kmeans"): the clustering algorithm to apply. Can be
            'kmeans' or 'gmm'
        transform (optional): function that maps waveforms to a new
            feature space

    Returns:
        A core.ClusterDataset object with one child for each cluster at
        each timestep.
    """
    _fn_start = time.time()

    _new_labels = -1 * np.ones(len(dataset)).astype(np.int)
    len_last_window = dpoints
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
        print(
            "Completed {}/{} in {:.1f}s.".format(
                np.max(next_window),
                len(dataset), time.time() - _fn_start
            ),
            end="\r")

    print("Completed clustering in {:.1f}s".format(time.time() - _fn_start))
    return dataset.cluster(_new_labels)


def reassign_unassigned(waveforms, labels):
    neigh = KNeighborsClassifier(n_neighbors=1)
    if len(np.where(labels != -1)[0]) == 0:
        return labels

    if len(np.where(labels == -1)[0]) == 0:
        return labels

    neigh.fit(waveforms[labels != -1], labels[labels != -1])
    labels[np.where(labels == -1)] = neigh.predict(
            waveforms[np.where(labels == -1)]
    )

    return labels


def denoise_step(
        dataset,
        current_node,
        min_waveforms,
        dt=None,
        dpoints=None,
        n_components=None,
        mode=None):
    """Perform clustering and then reassign data values"""
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
        dataset.waveforms[flat.ids] = [
                centroids[label] for label in flat.labels]

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
    for n in range(20, 4, -2):
        hdb = hdbscan.HDBSCAN(min_cluster_size=n)
        final_labels = hdb.fit_predict(tsned_with_time)
        if 20 <= len(np.unique(final_labels)) <= 40:
            print("Selected min_cluster_size={}".format(n))
            break

    final_labels = reassign_unassigned(tsned_with_time, final_labels)

    return denoised.cluster(final_labels)
