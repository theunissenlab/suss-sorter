"""
General way things go

1. Divide into windows
regular
2. KMeans overclustering on each window
3. Merging of clusters within window
isosplit
similarity metric

4. Reclustering across time
5. Merging of clusters across windows
isosplit
similarity metric
"""
import time

import hdbscan
import numpy as np
import scipy.stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import BayesianGaussianMixture

from .core import DataNode


def sort(
        times,
        waveforms,
        sample_rate,
        sparse_fn: "func mapping data to sparse representation",
        denoising_1_sparse: "use sparse encoding in first clustering step" = True,
        denoising_1_window: "denoising window (seconds)" = 60.0,
        n_denoising_1_clusters: "number of denoising clusters" = 25,
        denoising_2_window: "time window for second clustering step (seconds)" = 300.0,
        max_denoising_2_clusters: "max clusters for gaussian mixture in second clustering step" = 10,
        denoising_2_sparse: "use sparse encoding in second clustering step" = False, 
        denoising_2_pcs: "number of PCs to use in second clustering step" = 6,
        denoising_2_min_cluster_weight: "minimum weight of cluster during second step" = 0.02,
        denoising_2_min_cluster_size: "minimum size of cluster during second step" = 3,
        spacetime_sparse: "use sparse encoding in spacetime representation" = True,
        spacetime_pcs: "number of pcs to use for spacetime representation" = 3,
        hdb_min_cluster_size: "min cluster size for final hdbscan step" = 2,
        verbose = False
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

    master_node = DataNode(times=times, waveforms=waveforms, sample_rate=sample_rate)

    if verbose: print("Initializing...")

    # TODO (kevin): dont duplicate the sparse encoding
    if spacetime_sparse:
        spacetime_pca = PCA(n_components=spacetime_pcs).fit(sparse_fn(master_node.waveforms))
    else:
        spacetime_pca = PCA(n_components=spacetime_pcs).fit(master_node.waveforms)

    if verbose:
        print("Sorting {} waveforms ({:.1f} hours of data)".format(
            len(master_node), (np.max(master_node.times) - np.min(master_node.times)) / (60.0 * 60.0)))
    t_start = time.time()

    _denoised_nodes = []
    for window, _ in master_node.windows(dt=denoising_1_window):
        kmeans = KMeans(n_clusters=min(n_denoising_1_clusters, len(window.waveforms)))
        labels = kmeans.fit_predict(
                sparse_fn(window.waveforms)
                if denoising_1_sparse
                else windows.waveforms
        )
        _denoised_nodes += [window.select(labels == label) for label in np.unique(labels)]

    denoised_node = DataNode(children=_denoised_nodes)

    if verbose: print("First denoising step done in {:.1f}s. Reduced to {} clusters".format(time.time() - t_start, len(denoised_node)))
    t_start = time.time()

    _representative_nodes = []
    for window, _ in denoised_node.windows(dt=denoising_2_window):
        gmm = BayesianGaussianMixture(n_components=max_denoising_2_clusters, max_iter=500)
        _data = sparse_fn(window.waveforms) if denoising_2_sparse else window.waveforms
        _data = PCA(n_components=denoising_2_pcs).fit_transform(_data)
        gmm.fit(_data)

        labels = gmm.predict(_data)

        def _keep_label(label):
            return (
                (gmm.weights_[label] >= denoising_2_min_cluster_weight) and
                (np.sum(labels == label) >= denoising_2_min_cluster_size)
            )

        _representative_nodes += [
                window.select(labels == label)
                for label in np.unique(labels)
                if _keep_label(label)
        ]

    denoised_node = DataNode(children=_representative_nodes)

    if verbose: print("Second denoising step done in {:.1f}s. Reduced to {} clusters".format(time.time() - t_start, len(denoised_node)))
    t_start = time.time()

    spacetime_representation = np.hstack([
        spacetime_pca.transform(
            sparse_fn(denoised_node.waveforms) if spacetime_sparse else denoised_node.waveforms
        ),
        denoised_node.times[:, None]
    ])
    spacetime_representation = scipy.stats.zscore(spacetime_representation, axis=0)

    tsne = TSNE(n_components=2)
    spacetime_tsne = tsne.fit_transform(spacetime_representation)
    hdb = hdbscan.HDBSCAN(min_cluster_size=hdb_min_cluster_size)
    labels = hdb.fit_predict(spacetime_tsne)
    # TODO (kevin): rejected points (labeled -1) can be assigned to nearest cluster

    final_node = DataNode(children=[denoised_node.select(labels == label) for label in np.unique(labels)])

    if verbose: print("Final step done in {:.1f}s. Reduced to {} clusters".format(time.time() - t_start, len(final_node)))
    t_start = time.time()

    return final_node
