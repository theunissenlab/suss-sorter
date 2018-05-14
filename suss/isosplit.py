"""
Python implementation of the isosplit cluster merging algorithm for spike sorting

http://github.com/flatironinstitute/mountainsort

Does not do reassignment of multi-modal cluster pairs

Use like this:

>>> from spikes.sorting.merging import isosplit_cluster
>>> labels = isosplit_cluster(waveforms, threshold=1.0, k_initial=30)
"""

import numpy as np

from sklearn.cluster import KMeans


def kolgomorov_smirnov(a, b):
    """Largest distance between CDF of two distributions"""
    a_distribution = np.cumsum(a) / np.sum(a)
    b_distribution = np.cumsum(b) / np.sum(b)
                    
    diff = np.abs(a_distribution - b_distribution)

    return np.max(diff) * np.sqrt(np.mean([np.sum(a), np.sum(b)]))


def ks_modified(a, b, pivot):
    """Modified Kolomogorov-Smirnov test from MountainSort

    Looks near the edges of the distribution for areas that
    locally deviate from unimodality??

    Seems overly sensitive to edge scenarios
    """
    a_left = a[:pivot + 1]
    b_left = b[:pivot + 1]


    ks_best = 0.0
    crit_range = slice(0, None)

    if len(a_left):
        for div in np.arange(1, np.floor(np.log2(len(a_left))) - 1):
            length = int(len(a_left) // div)
            ks = kolgomorov_smirnov(a_left[:length], b_left[:length])
            if ks > ks_best:
                ks_best = ks
                crit_range = slice(0, length - 1)

    a_right = a[pivot + 1:][::-1]
    b_right = b[pivot + 1:][::-1]

    if len(a_right):
        for div in np.arange(1, np.floor(np.log2(len(a_right))) - 1):
            length = int(len(a_right) // div)
            ks = kolgomorov_smirnov(a_right[:length], b_right[:length])
            if ks > ks_best:
                ks_best = ks
                crit_range = slice(len(a_right) + pivot - length, None)

    return ks_best, crit_range


class IsoSplit(object):
    """IsoSplit algorithm from MountainSort"""

    def __init__(self, scale=1.0):
        self.scale = scale

    def isocut(self, X):
        X.sort()

        n_samples = len(X)

        n_bins = int(min(np.ceil(np.sqrt(n_samples / 2.0) * self.scale), np.sqrt(3 + n_samples) - 1))

        intervals = np.min([
            np.arange(1, n_bins + 1),
            np.arange(n_bins, 0, -1)
        ], axis=0).astype(np.float)

        intervals *= np.float(n_samples - 1) / np.sum(intervals)

        idx = np.zeros(len(intervals) + 1)
        for i in range(len(intervals)):
            idx[i + 1] = idx[i] + intervals[i]

        x_sub = X[idx.astype(int)]
        point_spacing = x_sub[1:] - x_sub[:-1] # np.diff(x_sub)
        bin_spacing = idx[1:] - idx[:-1] # np.diff(idx)
        density = bin_spacing / point_spacing  # estimated point density at each bin

        isotonic_fit = self.fit_isotonic_up_down(density, bin_spacing)

        residual = density - isotonic_fit
        weighted_fit = isotonic_fit * point_spacing

        peak = np.argmax(isotonic_fit)

        # the critical range is where we think the optimal cut might be
        ks_best, crit_range = ks_modified(bin_spacing, weighted_fit, peak)

        residual = residual[crit_range]
        weights = point_spacing[crit_range]

        fit_residual = self.fit_isotonic_down_up(residual, weights)

        cutpoint = np.argmin(fit_residual) + crit_range.start
        cutpoint_x = np.mean(x_sub[cutpoint:cutpoint + 2])

        return ks_best, cutpoint_x, x_sub[:-1] + point_spacing / 2, isotonic_fit, density

    def fit_isotonic(self, x, weights, N=None):
        """Fit a monotonically increasing or decreasing function"""
        if N is None:
            N = len(weights)
        
        counts = np.zeros(N)
        raw_counts = np.zeros(N)
        totals = np.zeros(N)
        totals_square = np.zeros(N)
        mse = np.zeros(N)
        fit = np.zeros(N)
        
        last_index = 0
        raw_counts[0] = 1
        counts[last_index] = weights[0]
        totals[last_index] = x[0] * weights[0]
        totals_square[last_index] = x[0] * x[0] * weights[0]
        
        for i in range(1, N):
            last_index += 1
            
            raw_counts[last_index] = 1
            counts[last_index] =  weights[i]
            totals[last_index] = x[i] * weights[i]
            totals_square[last_index] = x[i] * x[i] * weights[i]
            mse[i] = mse[i - 1]
                    
            while True:
                if last_index <= 0:
                    break
                if totals[last_index - 1] / counts[last_index - 1] < totals[last_index] / counts[last_index]:
                    break
                else:
                    previous_mse = totals_square[last_index - 1] - np.power(totals[last_index - 1], 2) / counts[last_index - 1]
                    previous_mse += totals_square[last_index] - np.power(totals[last_index], 2) / counts[last_index]
                    
                    raw_counts[last_index - 1] += raw_counts[last_index]
                    counts[last_index - 1] += counts[last_index]
                    totals[last_index - 1] += totals[last_index]
                    totals_square[last_index - 1] += totals_square[last_index]
                    mse[i] += totals_square[last_index - 1] - np.power(totals[last_index - 1], 2) / counts[last_index - 1]
                    last_index -= 1
        
        k = 0
        for i in range(last_index + 1):
            for j in range(int(raw_counts[i])):
                fit[k + j] = totals[i] / counts[i]
            k += int(raw_counts[i])
            
        return fit, mse

    def fit_isotonic_up_down(self, x, weights):
        """Best unimodal fit by optimally fitting an increasing and decreasing function
        """
        N = len(weights)
        
        x_rev = x[::-1]
        weights_rev = weights[::-1]
        
        fitted, mse = self.fit_isotonic(x, weights)
        fitted_rev, mse_rev = self.fit_isotonic(x_rev, weights_rev)
        
        mse += mse_rev[::-1]
        best_ind = np.argmin(mse)
        best_val = mse[best_ind]
        fitted, mse = self.fit_isotonic(x, weights, N=best_ind + 1)
        fitted_rev, mse_rev = self.fit_isotonic(x_rev, weights_rev, N=N-best_ind)
        return np.concatenate([fitted, fitted_rev[-2::-1]])

    def fit_isotonic_down_up(self, x, weights):
        x_neg = -x
        return -self.fit_isotonic_up_down(x_neg, weights)


def compute_means(data, labels):
    return dict(
        (label, np.mean(data[labels == label], axis=0)) for label in np.unique(labels)
    )


def dist(label1, label2, means=None):
    try:
        return np.linalg.norm(means[label2] - means[label1])
    except:
        return np.inf


class Tracker(object):
    def __init__(self, unique_labels):
        self.unique_labels = unique_labels
        self.checked = []
    
    def pairs(self, sorter=None):
        pairs = [
            {label1, label2}
            for label1 in self.unique_labels
            for label2 in self.unique_labels
            if label1 < label2
            and set([label1, label2]) not in self.checked
        ]
        if sorter is not None:
            return sorted(pairs, key=lambda x: dist(*list(x), means=sorter))
        else:
            return pairs
        
    def touch(self, label):
        self.checked = list(filter(lambda x: label not in x, self.checked))
        
    def remove(self, label):
        self.unique_labels = list(filter(lambda x: x != label, self.unique_labels))
    
    def check(self, pair):
        # free up this to be checked again
        self.checked.append(set(pair))


class Merger(object):

    def __init__(self, labels, initial_threshold, final_threshold, mode="linear"):
        self.n_initial = len(np.unique(labels))
        self.n_initial = 3 if self.n_initial <= 3 else self.n_initial
        self.update(self.n_initial)
        self._map = self._generate_linear_map(initial_threshold, final_threshold)

    def _generate_linear_map(self, start, stop):
        return lambda n: stop + (start - stop) * ((n - 2.) / (self.n_initial - 2.))

    def update(self, cluster_count):
        self.n_current = cluster_count

    @property
    def threshold(self):
        return self._map(self.n_current)

    def is_unimodal(self, k):
        if k < self.threshold:
            return True
        else:
            return False

