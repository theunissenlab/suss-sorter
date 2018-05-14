"""
Cluster evaluation and metrics
"""

import numpy as np

import scipy.stats
from sklearn.neighbors import NearestNeighbors

from .core import ClusterDataset
from .isosplit import IsoSplit


def unimodality(cluster1, cluster2):
    """Measures how well poorly two clusters are described by a unimodal distribution

    Uses Isosplit algorithm (Magland 2015)
    """
    raise NotImplementedError


def isi_histogram(cluster, bins, range):
    """Generate a historgram of interspike intervals"""
    return np.histogram(np.diff(cluster.times), bins=bins, range=range)


def isi_violations(cluster):
    """Interspike interval violations
    
    Return both absolute count and fraction of events
    """
    isi_violations = np.sum(np.diff(cluster.times) < 0.001)
    return isi_violations, isi_violations / len(cluster.times)


def peak_distribution(cluster):
    """Distribution of peak values

    One aspect of cluster quality is that this distribution not extend below the detection threshold
    """
    return np.histogram(np.min(cluster.waveforms, axis=1))


def peak_skew(cluster, peak_idx=None):
    if peak_idx is None:
        peak_idx = cluster.waveforms.shape[1] / 2

    peaks = cluster.waveforms.T[peak_idx]
    return np.mean(((peaks - np.mean(peaks)) / np.std(peaks)) ** 3)


def residuals(cluster):
    """Compute mean and std of cluster waveforms at each sample
    """
    mean = np.mean(cluster.waveforms, axis=0)
    std = np.std(cluster.waveforms, axis=0)
    return mean, std


def firing_rate(cluster, windows=0.01):
    """Compute baseline firing rate and max firing rate
    """
    raise NotImplementedError


def false_positives_refractory(cluster, refractory_period=0.001, censored_period=3.0/30000.0):
    n_violations = isi_violations(cluster)[0]
    n_spikes = len(cluster)
    T = np.max(cluster.times) - np.min(cluster.times)
    '''
    n_spikes = 10000
    T = 1
    refractory_period = 0.003
    censored_period = 0.001
    n_violations = 20
    T = 1000
    '''
    X = (T * n_violations) / (2 * (refractory_period - censored_period) * pow(n_spikes, 2))
    print("T", T)
    print("N", n_spikes)
    print("r", n_violations)
    print("t-t", refractory_period - censored_period)
    # print(X)
    return (1.0 - np.sqrt(1 - 4 * X)) / 2.0


def false_negatives_peak_distribution(cluster):
    raise NotImplementedError


def false_positives_gaussian_overlap(cluster, others):
    """Gaussian mixture model overlapping but need to consider changes over time"""
    raise NotImplementedError


def false_negatives_censored(cluster, others, censored_period=3.0/30000.0):
    """How many spikes we expect to have missed because of other spike arrivals"""
    # TODO (kevin): the unit may appear and disappear over time... so this could be an overestimate
    return len(others) * censored_period / (np.max(cluster.times) - np.min(cluster.times))


def _compute_neighbor_labels(joined, k=5):
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(joined.waveforms)
    neighbors = nn.kneighbors(joined.waveforms, n_neighbors=k, return_distance=False)

    # compute fraction of k neighbors that belong to the correct cluster
    return np.vstack([
        (k_col == joined.labels).astype(np.int)
        for k_col in joined.labels[neighbors].T
    ])


def isolation(*clusters, k=5, pairwise=False):
    """Isolation metric used in Mountainsort paper
    """
    if pairwise:
        final_labels = np.arange(len(clusters))
        iso = np.ones((len(clusters), len(clusters)))
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                joined = ClusterDataset([clusters[a], clusters[b]])
                joined = joined.flatten(assign_labels=True)

                pairwise_isolation = (
                    np.sum(_compute_neighbor_labels(joined, k=k)) /
                    (k * len(joined))
                )
                iso[a, b] = iso[b, a] = pairwise_isolation

    else:
        joined = ClusterDataset(clusters)
        joined = joined.flatten(assign_labels=True)
        final_labels = np.unique(joined.labels)

        _overlap = _compute_neighbor_labels(joined, k=k)
        iso = np.array([
            np.mean(np.mean(_overlap, axis=0)[joined.labels == label])
            for label in final_labels
        ])

    return final_labels, iso


def isolation_overlaps(dataset, dt):
    isolations = []
    t_arr = []
    for t, _, window in dataset.windows(dt=dt):
        isolations.append(isolation(*window.nodes, pairwise=True))
        t_arr.append(t)
        yield t_arr[-1], isolations[-1]
    # return np.array(t_arr), np.array(isolations)


def noise_overlap(cluster):
    """Noise overlap metric used in Mountainsort paper

    Requires randomly selected data from raw signal
    """
    raise NotImplementedError
