"""
Cluster evaluation and metrics
"""

import numpy as np

from .isosplit import IsoSplit


def unimodality(node1, node2):
    """Measures how well poorly two nodes are described by a unimodal distribution

    Uses Isosplit algorithm (Magland 2015)
    """
    raise NotImplementedError


def isi_histogram(node, bins, range):
    """Generate a historgram of interspike intervals"""
    return np.histogram(np.diff(node.times), bins=bins, range=range)


def isi_violations(node):
    """Interspike interval violations
    
    Return both absolute count and fraction of events
    """
    isi_violations = np.sum(np.diff(node.times) < 0.001)
    return isi_violations, isi_violations / len(node.times)


def peak_distribution(node):
    """Distribution of peak values

    One aspect of cluster quality is that this distribution not extend below the detection threshold
    """
    return np.histogram(np.min(node.waveforms, axis=1))


def residuals(node):
    """Compute mean and std of node waveforms at each sample
    """
    mean = np.mean(node.waveforms, axis=0)
    std = np.std(node.waveforms, axis=0)
    return mean, std


def firing_rate(node, windows=0.01):
    """Compute baseline firing rate and max firing rate
    """
    raise NotImplementedError


def false_positives_refractory(node, refractory_period=0.001, censored_period=3.0/30000.0):
    n_violations = isi_violations(node)[0]
    n_spikes = len(node.times)
    X = n_violations / (2 * (refractory_period - censored_period) * pow(n_spikes, 2))
    return (1 + np.sqrt(1 - 4 * X)) / 2


def false_negatives_peak_distribution(node):
    raise NotImplementedError


def false_positives_gaussian_overlap(node, others):
    """Gaussian mixture model overlapping but need to consider changes over time"""
    raise NotImplementedError


def false_negatives_censored(node, others):
    """How many spikes we expect to have missed because of other spike arrivals"""
    raise NotImplementedError


def isolation(node, others):
    """Isolation metric used in Mountainsort paper
    """
    raise NotImplementedError


def noise_overlap(node):
    """Noise overlap metric used in Mountainsort paper

    Requires randomly selected data from raw signal
    """
    raise NotImplementedError
