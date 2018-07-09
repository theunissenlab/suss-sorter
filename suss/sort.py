import time

import hdbscan
import numpy as np
import scipy.stats
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

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
        min_cluster_size=10,
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
            if count < min_cluster_size:
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
        min_cluster_size=min_waveforms,
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
        dict(min_waveforms=20, dpoints=1000, n_components=30, mode="kmeans"),
        # dict(min_waveforms=30, dpoints=1000, n_components=20, mode="kmeans"),
        # dict(min_waveforms=15, dpoints=2000, n_components=16, mode="kmeans"),
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


def isi(node):
    dt = np.diff(node.flatten().times)
    return np.sum(dt < 0.001) / len(dt)


def cluster_quality(data, labels, n_neighbors=20):
    neighbors = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="ball_tree"
    ).fit(data)

    _, indices = neighbors.kneighbors(data)
    quality = {}
    for label in np.unique(labels):
        cluster_size = len(np.where(labels == label)[0])
        take_n = min(n_neighbors, cluster_size)
        neighbor_idx = indices[labels == label, 1:take_n]
        has_bad_neighbor = np.any(labels[neighbor_idx] == label, axis=1)
        quality[label] = {
            "count": cluster_size,
            "isolation": np.mean(has_bad_neighbor)
        }

    return quality


def get_flippable_points(data, labels, n_neighbors=10):
    neighbors = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="ball_tree"
    ).fit(data)

    _, indices = neighbors.kneighbors(data)
    return np.array([
        np.mean(labels[indices[idx, 1:]] != label) > 0.5
        for idx, label in enumerate(labels)
    ])


def cleanup_clusters(data, labels, n_neighbors=20):
    cleaner = KNeighborsClassifier(n_neighbors=n_neighbors)
    cleaner.fit(data, labels)
    labels = cleaner.predict(data)
    return labels


def flip_points(data, labels, flippable, n_neighbors=10, create_labels=False):
    if len(flippable) == 0:
        return labels
    if create_labels:
        hdb = hdbscan.HDBSCAN(min_cluster_size=3)
        potential_labels = hdb.fit_predict(data[flippable])
        if -1 in potential_labels:
            potential_labels = reassign_unassigned(
                    data[flippable],
                    potential_labels)
        labels[flippable] = np.max(labels) + potential_labels + 1
        return cleanup_clusters(data, labels, n_neighbors=n_neighbors)
    else:
        replacer = KNeighborsClassifier(n_neighbors=n_neighbors)
        replacer.fit(data, labels)
        replacement_probs = replacer.predict_proba(data[flippable])

        _flippable_classes = np.isin(replacer.classes_, np.unique(labels[flippable]))
        classes = replacer.classes_[_flippable_classes]
        replacement_labels = classes[
            np.argmax(replacement_probs[:, _flippable_classes], axis=1)
        ]
        labels[flippable] = replacement_labels
        return labels


def tsne_time(dataset, perplexity=30, t_scale=30 * 60 * 60, pcs=12):
    tsned = TSNE(n_components=2, perplexity=perplexity).fit_transform(
        PCA(n_components=pcs).fit_transform(dataset.waveforms)
    )
    wf_arr = scipy.stats.zscore(tsned, axis=0)
    t_arr = dataset.times / t_scale
    t_arr = t_arr - np.mean(t_arr)

    return TSNE(n_components=2, perplexity=perplexity).fit_transform(
        np.hstack([wf_arr, t_arr[:, None]])
    )


def pca_time(dataset, t_scale=30 * 60 * 60, pcs=6):
    pcaed = PCA(n_components=pcs).fit_transform(dataset.waveforms)
    wf_arr = scipy.stats.zscore(pcaed, axis=0)
    t_arr = dataset.times / t_scale
    t_arr = t_arr - np.mean(t_arr)

    return PCA(n_components=pcs).fit_transform(
        np.hstack([wf_arr, t_arr[:, None]])
    )


def is_isolated(labels, quality_dict, min_count=12, min_isolation=0.99):
    return np.array([
        (
            (min_count <= quality_dict[label]["count"]) and
            (min_isolation <= quality_dict[label]["isolation"])
        ) for label in labels
    ])


def whittle(dataset, n=10):
    """Whittle down a dataset to find isolated clusters
    """
    temp_dataset = dataset
    final_labels = -1 * np.ones(len(dataset)).astype(np.int)
    for idx in range(n):
        mask = final_labels == -1

        if np.sum(mask) < 100:
            break

        temp_dataset = dataset.select(mask)
        tsned = tsne_time(temp_dataset)

        hdb = hdbscan.HDBSCAN(min_cluster_size=3)
        labels = hdb.fit_predict(tsned)
        if -1 in labels:
            labels = reassign_unassigned(tsned, labels)

        quality = cluster_quality(tsned, labels)
        isolated = is_isolated(labels, quality)
        labels[np.logical_not(isolated)] = -1
        labels[labels != -1] += np.max(final_labels) + 1
        final_labels[mask] = labels

    result = dataset.cluster(final_labels)
    return result.select(
            [isi(n) < 0.05 for n in result.nodes],
            child=False)


def denoise(times, waveforms):
    denoised = denoising_sort(times, waveforms)
    denoised = denoised.select([isi(n) < 0.05 for n in denoised.nodes])

    denoised = cluster_step(
        denoised,
        dpoints=200,
        n_components=20,
        min_cluster_size=5,
        mode="kmeans"
    )
    denoised = denoised.select([isi(n) < 0.05 for n in denoised.nodes])
    return denoised


def sort(denoised):
    whittled = whittle(denoised)

    flat = whittled.flatten(1)
    tsned = tsne_time(flat)
    labels = flat.labels

    labels = cleanup_clusters(tsned, labels, n_neighbors=20)
    labels = cleanup_clusters(tsned, labels, n_neighbors=10)

    # Not sure if the next few lines help
    flippable = get_flippable_points(tsned, labels)
    labels = flip_points(tsned, labels, flippable, create_labels=True)
    labels = cleanup_clusters(tsned, labels, n_neighbors=20)

    counts = dict(
            (label, np.sum(labels == label))
            for label in np.unique(labels)
    )
    flippable = [counts[label] < 10 for label in labels]
    if np.any(flippable):
        labels = flip_points(tsned, labels, flippable)
    labels = cleanup_clusters(tsned, labels, n_neighbors=20)

    return tsned, flat.cluster(labels)
