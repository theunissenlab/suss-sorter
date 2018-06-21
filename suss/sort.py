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
    print("Completed clustering of {:.2f} min in {:.1f}s.".format(
        (t_start + dt) / 60.0, time.time() - _fn_start))

    return ClusterDataset(np.concatenate(_denoised_nodes))


def space_time_transform(node, transform=None, zscore=True,
        waveform_features=3, time_features=True, perplexity=30.0):
    """Transform data into highly separable representation using T-SNE

    Args:
        node: An instance of core.BaseDataset containing the waveforms to be clustered
        transform (optional): A function mapping waveforms to a new feature space
            that will be joined with temporal data
        waveform_features (default: 3): An integer number of waveform features
            to include in the T-SNE fitting
        time_features (default: True): A boolean indicator for whether to
            join the spike time as a feature

    Returns:
        A numpy array of shape (n_spikes, 2) of the data in two dimensions using
        the T-SNE transform.
    """
    data = transform(node.waveforms) if transform is not None else node.waveforms
    if not len(data):
        return data

    if data.shape[1] >= waveform_features:
        if len(node.nodes) > waveform_features:
            lda  = LDA(n_components=waveform_features).fit(
                    node.flatten(assign_labels=True).waveforms,
                    node.flatten(assign_labels=True).labels
            )
            data = lda.transform(data)
        else:
            data = PCA(n_components=waveform_features).fit_transform(data)

    # Incorporate time into the spike representation
    # TODO (kevin): Weight of time feature should be a parameter
    if time_features:
        space_time = np.hstack([data, node.times[:, None]])
    else:
        space_time = data

    if zscore:
        space_time = scipy.stats.zscore(space_time, axis=0)

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000)
    return tsne.fit_transform(space_time)


def prune(dataset, min_cluster_size=5):
    return dataset.select([len(cluster) >= min_cluster_size for cluster in dataset.nodes])


def default_sort(times, waveforms, sample_rate, sparse_fn=None):
    """Sort function with 'default' parameters"""
    spike_dataset = SpikeDataset(times=times, waveforms=waveforms, sample_rate=sample_rate)

    denoised_clusters = cluster_step(spike_dataset,
            dt=0.5 * 60.0,
            n_components=25,
            mode="kmeans",
            transform=None)
    denoised_clusters = prune(denoised_clusters, 5)

    clustered_clusters = cluster_step(denoised_clusters,
            dt=5 * 60.0,
            mode="tsne-dbscan",
            transform=lambda data: PCA(n_components=10).fit_transform(data)
    )
    clustered_clusters = prune(clustered_clusters, 2)

    # could try this twice
    space_time = space_time_transform(
        clustered_clusters,
        transform=None,
        zscore=True,
        waveform_features=1,
        time_features=True,
        perplexity=30.0,
    )

    hdb = hdbscan.HDBSCAN(min_cluster_size=10)
    labels = hdb.fit_predict(space_time)

    result = clustered_clusters.cluster(labels)

    return space_time, labels, result


def reassign_unassigned(waveforms, labels):
    neigh = KNeighborsClassifier(n_neighbors=1)
    if len(np.where(labels != -1)[0]) == 0:
        return labels

    neigh.fit(waveforms[labels != -1], labels[labels != -1])
    labels[np.where(labels == -1)] = neigh.predict(waveforms[np.where(labels == -1)])
    return labels


def sexy_sort(times, waveforms, sample_rate, sparse_fn=None):
    """Sort function with 'default' parameters"""
    spike_dataset = SpikeDataset(times=times, waveforms=waveforms, sample_rate=sample_rate)

    denoised_clusters = cluster_step(spike_dataset,
            dt=0.5 * 60.0,
            n_components=30,
            mode="kmeans",
            transform=None)
    denoised_clusters = prune(denoised_clusters, 5)

    clustered_clusters = cluster_step(denoised_clusters,
            dt=50 * 60.0,
            n_components=30,
            mode="kmeans",
            transform=None
    )
    clustered_clusters = prune(clustered_clusters, 2)

    tsned = space_time_transform(
        clustered_clusters,
        transform=None,
        waveform_features=20,
        zscore=False,
        time_features=False,
        perplexity=30.0,
    )

    if len(tsned):
        hdb = hdbscan.HDBSCAN(min_cluster_size=2)
        labels = hdb.fit_predict(tsned)
        if -1 in np.unique(labels):
            labels = reassign_unassigned(clustered_clusters.waveforms, labels)
    else:
        labels = []

    result = clustered_clusters.cluster(labels)

    return tsned, labels, result


def denoise_step(dataset, current_node, min_waveforms, dt=None, dpoints=None, n_components=None, mode=None):
    denoised_node = cluster_step(
        current_node.flatten(),
        dt=dt,
        dpoints=dpoints,
        n_components=n_components,
        mode=mode
    )

    mask = [node.waveform_count >= min_waveforms for node in denoised_node.nodes]
    denoised_node = denoised_node.select(mask)

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

    return mask, denoised_node


def denoising_sort(times, waveforms, sample_rate):
    spike_dataset = SpikeDataset(times=times, waveforms=waveforms, sample_rate=sample_rate)

    original_waveforms = spike_dataset.waveforms.copy()

    steps = [
        dict(min_waveforms=2, dt=0.5 * 60.0, n_components=32, mode="kmeans"),
        dict(min_waveforms=15, dt=1.0 * 60.0, n_components=16, mode="kmeans"),
        dict(min_waveforms=20, dt=1.0 * 60.0, n_components=10, mode="kmeans"),
        dict(min_waveforms=20, dt=60.0 * 60.0, n_components=12, mode="spectral"),
    ]
    steps = [
        dict(min_waveforms=2, dpoints=1000, n_components=32, mode="kmeans"),
        dict(min_waveforms=15, dpoints=2000, n_components=16, mode="kmeans"),
        dict(min_waveforms=20, dpoints=2000, n_components=10, mode="kmeans"),
    ]


    dataset = spike_dataset
    denoised_node = dataset
    try:
        for step_kwargs in steps:
            mask, denoised_node = denoise_step(dataset, denoised_node, **step_kwargs)
    except:
        raise
    finally:
        spike_dataset.waveforms[:] = original_waveforms

    # flat = denoised_node.flatten(assign_labels=True)
    return denoised_node
    # return dataset.select(flat.ids).cluster(flat.labels)


def sort_v3(times, waveforms, sample_rate):
    denoised = denoising_sort(times, waveforms, sample_rate)

    tsne = TSNE(n_components=2, perplexity=10)
    manifold = tsne.fit_transform(denoised.waveforms)

    labels = np.zeros(len(denoised.nodes))
    n_labels_per_window = 12

    spectral = SpectralClustering(n_clusters=n_labels_per_window)
    spectral = hdbscan.HDBSCAN(min_cluster_size=3)

    for window_idx, (t_start, t_stop, window) in enumerate(denoised.windows(dt=2 * 60.0 * 60.0)):
        selector = np.where(
            (denoised.times >= t_start) &
            (denoised.times < t_stop)
        )[0]
        wf_features = scipy.stats.zscore(manifold[selector], axis=0)
        time_features = window.times[:, None] / np.max(denoised.times)

        stacked = np.hstack([wf_features, time_features])

        new_labels = spectral.fit_predict(
                stacked
        )
        if -1 in np.unique(new_labels):
            new_labels = reassign_unassigned(stacked, new_labels)

        labels[selector] = new_labels + np.max(labels) + 1

    return denoised.cluster(labels)


def sort(
        times,
        waveforms,
        sample_rate,
        sparse_fn: "func mapping data to sparse representation",
        denoising_1_sparse: "use sparse encoding in first clustering step" = True,
        denoising_1_window: "denoising window (seconds)" = 60.0,
        denoising_1_min_cluster_size: "minimum cluster size during first step" = 5,
        n_denoising_1_clusters: "number of denoising clusters" = 25,
        denoising_2_window: "time window for second clustering step (seconds)" = 300.0,
        max_denoising_2_clusters: "max clusters for gaussian mixture in second clustering step" = 10,
        denoising_2_sparse: "use sparse encoding in second clustering step" = False, 
        denoising_2_pcs: "number of PCs to use in second clustering step" = 6,
        denoising_2_min_cluster_size: "minimum size of cluster during second step" = 4,
        spacetime_sparse: "use sparse encoding in spacetime representation" = True,
        spacetime_pcs: "number of pcs to use for spacetime representation" = 3,
        hdb_min_cluster_size: "min cluster size for final hdbscan step" = 2,
        perplexity: "perplexity parameter for t-sne on last cluster step" = 10.0,
        verbose = False,
        debug = False
    ):
    """Run sorting algorithm on set of spike waveforms and arrival times

    1. Denoising clustering - overclustering with k-means
        (params: window_size, n_clusters, feature_space [sparse, pcs, raw])
    2. Denoising clustering 2 - bayesian gaussian mixture to generate candidate clusters
        (params: window_size, max_clusters, feature_space, 
                 min_cluster_size, min_cluster_weight)
    3. Spacetime transform - encoding representative clusters and time information
            in TSNE manifold
        (params: feature_space)
    4. Spacetime clustering - clustering in spacetime space using hierarchical
            density based clustering HDBSCAN
        (params: min_cluster_size)
    5. Final cluster merging - merging clusters by unimodality, similar waveforms,
            proximity in time, ISI violations, firing properties, etc...
    """

    spike_dataset = SpikeDataset(times=times, waveforms=waveforms, sample_rate=sample_rate)

    if verbose: print("Initializing...")

    # TODO (kevin): dont duplicate the sparse encoding
    if spacetime_sparse:
        spacetime_pca = PCA(n_components=spacetime_pcs).fit(sparse_fn(spike_dataset.waveforms))
    else:
        spacetime_pca = PCA(n_components=spacetime_pcs).fit(spike_dataset.waveforms)

    if verbose:
        print("Sorting {} waveforms ({:.1f} hours of data)".format(
            len(spike_dataset.times),
            (np.max(spike_dataset.times) - np.min(spike_dataset.times)) / (60.0 * 60.0)
        ))
    t_start = time.time()

    denoised_node = cluster_step(
            spike_dataset,
            dt=denoising_1_window,
            n_components=n_denoising_1_clusters,
            mode="kmeans",
            transform=sparse_fn)

    denoised_node = prune(denoised_node, min_cluster_size=denoising_1_min_cluster_size)

    if verbose:
        print("First denoising step done in {:.1f}s. "
            "Reduced to {} clusters".format(time.time() - t_start, len(denoised_node)))
    t_start = time.time()

    # this step is different becuase tiny clusters are not rejected yet
    denoised_node = cluster_step(
            denoised_node,
            dt=denoising_2_window,
            n_components=max_denoising_2_clusters,
            mode="gmm",
            transform=lambda data: PCA(n_components=denoising_2_pcs).fit_transform(data))

    denoised_node = prune(denoised_node, min_cluster_size=denoising_2_min_cluster_size)

    # TODO (kevin): here we could have a step that removes outliers from each cluster?

    if verbose:
        print("Second denoising step done in {:.1f}s. "
            "Reduced to {} clusters".format(time.time() - t_start, len(denoised_node)))
    t_start = time.time()

    space_time = space_time_transform(
            denoised_node,
            transform=None,
            zscore=True,
            waveform_features=spacetime_pcs,
            time_features=True,
            perplexity=perplexity)

    hdb = hdbscan.HDBSCAN(min_cluster_size=hdb_min_cluster_size)
    labels = hdb.fit_predict(space_time)

    if debug:
        fig = plt.figure(figsize=(6, 6))
        for label in np.unique(labels):
            plt.scatter(*space_time[labels == label].T)
        plt.show()

    # TODO (kevin): rejected points (labeled -1) can be assigned to nearest cluster
    final_node = denoised_node.cluster(labels)

    if verbose:
        print("Final step done in {:.1f}s. "
            "Reduced to {} clusters".format(time.time() - t_start, len(final_node)))
    t_start = time.time()

    return final_node
