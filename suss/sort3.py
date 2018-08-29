import time

import networkx as nx
import numpy as np
import scipy.stats
import umap
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture

from .core import SpikeDataset, ClusterDataset
from .sort import SPC


def _compute_overlap(neighbors, labels, A, B):
    a_or_b = (labels == A) | (labels == B)

    size = np.sum(a_or_b)

    total = []

    sample_count = np.min([500, np.sum(labels==A), np.sum(labels==B)])
    if sample_count < 20:
        return 1.0
    for cluster in [A, B]:
        points = np.random.choice(np.arange(np.sum(labels == cluster)), size=sample_count, replace=False)

        for pt in points:
            k_nearest_labels = labels[neighbors[labels == cluster][pt]]
            total.append(np.mean(k_nearest_labels == cluster))

    return np.mean(total)


def isolation(waveforms, labels, k=10):
    unique_labels = np.unique(labels)
    overlaps = np.zeros((len(unique_labels), len(unique_labels)))

    knn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', n_jobs=2).fit(waveforms)
    _, neighbors = knn.kneighbors(waveforms)

    for i, A in enumerate(unique_labels):
        for j, B in enumerate(unique_labels):
            if i == j:
                overlaps[i, j] = 1.0
                continue
            overlaps[i, j] = _compute_overlap(neighbors, labels, A, B)

    mins = np.min(overlaps, axis=0)

    result = {}
    for i, label in enumerate(unique_labels):
        result[label] = mins[i]

    return result


def compute_skew(peaks):
    mean_peak = np.mean(peaks)
    return np.mean((peaks - mean_peak) ** 3 / np.std(peaks) ** 3)


def relabel(dataset, n_components=4):
    if not len(dataset):
        return np.array([])

    new_labels = -1 * np.ones(len(dataset.labels))
    for l in np.unique(dataset.labels):
        gmm = BayesianGaussianMixture(n_components=n_components,
                                      weight_concentration_prior=1 / (n_components * 2),
                                      max_iter=200)
        ttt = PCA(n_components=6).fit_transform(dataset.waveforms[dataset.labels == l])
        gmm_labels = gmm.fit_predict(ttt)
        new_labels[dataset.labels == l] = gmm_labels + 1 + np.max(new_labels)
    return new_labels


def umap_time(dataset, pcs, n_components=3, t_scale=(60.0 * 60.0),
        wf_start=0,
        wf_end=None):

    # FIXME: probably doesnt work with 1 datapoint...
    pcs = min(pcs, min(*dataset.waveforms.shape) - 1)
    n_components = min(n_components, len(dataset.waveforms) - 2)

    wf_slice = slice(wf_start, wf_end)
    return umap.UMAP(n_components=n_components).fit_transform(
        np.hstack([
            dataset.times[:, None] / t_scale,
            PCA(n_components=pcs).fit_transform(scipy.stats.zscore(dataset.waveforms[:, wf_slice], axis=0))
        ])
    )

def vote_on_labels(dataset, threshold=1.0):
    features = umap_time(dataset, pcs=12, n_components=6, t_scale=(60.0 * 60.0))
    spc = SPC(n_neighbors=min(5, len(dataset) // 2))
    spc.fit(features)
    result = spc.create_hierarchy()
    result = spc.collapse(result, threshold=threshold)
    labels = result.labels()

    return labels

import hdbscan

def vote_on_labels_hdb(dataset, min_cluster_size=10):
    features = umap_time(dataset, pcs=12, n_components=6, t_scale=(10.0 * 60.0), wf_start=10, wf_end=31)
    hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = hdb.fit_predict(features)
    cleaner = KNeighborsClassifier(n_neighbors=20).fit(features[labels != -1], labels[labels != -1])
    labels = cleaner.predict(features)
    cleaner = KNeighborsClassifier(n_neighbors=10).fit(features, labels)
    labels = cleaner.predict(features)

    return labels


def spc_clustering(dataset, threshold=1.0, repeat=5):
    if len(dataset) == 0:
        return np.array([])

    votes = []
    for _ in range(repeat):
        votes.append(vote_on_labels(dataset, threshold=threshold))

    label_map = {}
    next_label = 0
    labels = np.zeros_like(votes[0])
    for idx, label_key in enumerate(zip(*votes)):
        key = tuple(label_key)
        if key not in label_map:
            label_map[key] = next_label
            next_label += 1
        labels[idx] = label_map[key]
    return labels


def eliminate_small_clusters(dataset, labels, mode="high_snr"):
    if not len(dataset):
        return np.array([])

    if mode == "high_snr":
        t_scale = (2 * 60.0 * 60.0)
        wf_start = 0
        wf_end = None
        real_min_cluster_size = 1000
    elif mode == "low_snr":
        t_scale = (10.0 * 60.0)
        wf_start = (dataset.waveforms.shape[1] // 2) - 10
        wf_end = (dataset.waveforms.shape[1] // 2) + 11
        real_min_cluster_size = 10000

    features = umap_time(dataset, pcs=12, n_components=6,
            t_scale=t_scale, wf_start=wf_start, wf_end=wf_end)

    # First pass
    clustered = dataset.cluster(labels)
    solid_labels = []
    for label, node in clustered.labeled_nodes:
        if node.count >= real_min_cluster_size / 4.0:
            solid_labels.append(label)

    if len(solid_labels) and len(solid_labels) != len(np.unique(labels)):
        cleaner = KNeighborsClassifier(n_neighbors=5).fit(
            features[np.isin(labels, solid_labels)],
            labels[np.isin(labels, solid_labels)]
        )
        labels[np.logical_not(np.isin(labels, solid_labels))] = cleaner.predict(features)[np.logical_not(np.isin(labels, solid_labels))]

    # Second pass
    clustered = dataset.cluster(labels)
    solid_labels = []
    for label, node in clustered.labeled_nodes:
        if node.count >= real_min_cluster_size / 2.0:
            solid_labels.append(label)

    if len(solid_labels) and len(solid_labels) != len(np.unique(labels)):
        cleaner = KNeighborsClassifier(n_neighbors=5).fit(
            features[np.isin(labels, solid_labels)],
            labels[np.isin(labels, solid_labels)]
        )
        labels[np.logical_not(np.isin(labels, solid_labels))] = cleaner.predict(features)[np.logical_not(np.isin(labels, solid_labels))]

    # third pass
    clustered = dataset.cluster(labels)
    solid_labels = []
    for label, node in clustered.labeled_nodes:
        if node.count >= real_min_cluster_size:
            solid_labels.append(label)

    if len(solid_labels) and len(solid_labels) != len(np.unique(labels)):
        cleaner = KNeighborsClassifier(n_neighbors=5).fit(
            features[np.isin(labels, solid_labels)],
            labels[np.isin(labels, solid_labels)]
        )
        labels = cleaner.predict(features)

    return labels


def hdb_clustering(dataset, min_cluster_size=10, real_min_cluster_size=1000, repeat=5):
    if len(dataset) == 0:
        return np.array([])

    votes = []
    for _ in range(repeat):
        votes.append(vote_on_labels_hdb(dataset, min_cluster_size=10))

    label_map = {}
    next_label = 0
    labels = np.zeros_like(votes[0])
    for idx, label_key in enumerate(zip(*votes)):
        key = tuple(label_key)
        if key not in label_map:
            label_map[key] = next_label
            next_label += 1
        labels[idx] = label_map[key]

    labels = eliminate_small_clusters(dataset, labels, mode="low_snr")

    return labels



class SplitDataset(object):
    """Helper to manage hierarchical clustering two subsets of a dataset separately
    
    (e.g. upgoing and downgoing)
    """
    def __init__(self, dataset, condition):
        self.dataset = dataset
        self.level = 0
        
        self.set_1 = dataset.select(condition)
        self.set_2 = dataset.select(np.logical_not(condition))
        self.base_set_1 = self.set_1
        self.base_set_2 = self.set_2
    
    def __repr__(self):
        return "Set 1: {}\nSet 2: {}".format(str(self.set_1), str(self.set_2))
    
    def cluster(self, set_1_labels, set_2_labels):
        self.set_1 = self.set_1.cluster(set_1_labels)
        self.set_2 = self.set_2.cluster(set_2_labels)
        self.level += 1
        
    def flatten(self, n=None):
        if n is not None and n > self.level:
            raise Exception("Cannot flatten more than {} times".format(self.level))
            
        self.set_1 = self.set_1.flatten(n)
        self.set_2 = self.set_2.flatten(n)
        if n is None:
            self.level = 0
        else:
            self.level -= n
        
    def recombine(self):
        # base_labels = np.zeros(len(self.dataset))
        base_ids = self.dataset.ids
        base_labels = np.zeros(np.max(base_ids) + 1)
        flat_1 = self.set_1.flatten(self.level)
        flat_2 = self.set_2.flatten(self.level)
        if len(flat_1):
            base_labels[flat_1.ids] = flat_1.labels
        if len(flat_2):
            if len(flat_1):
                start = np.max(flat_1.labels)
            else:
                start = 0
            base_labels[flat_2.ids] = start + 1 - np.min(flat_2.labels) + flat_2.labels
        
        return self.dataset.cluster(base_labels[base_ids])
    
    def _skip(self, size, n):
        return max(1, size // n)
    
    def skip_1(self, n):
        return self._skip(len(self.set_1), n)
    
    def skip_2(self, n):
        return self._skip(len(self.set_2), n)
    
    def skip(self, n):
        return self._skip(len(self.dataset), n)


def cluster_step(
        dataset,
        dpoints=None,
        n_components=2,
        mode="kmeans",
        min_cluster_size=10,
        levels=4
    ):
    """Implement a first step of the hierarchical clustering algorithm

    Args:
        dataset: An (sub)instance of core.BaseDataset whose waveforms
            will be clustered
        dpoints: Number of points to take in each cluster step
        n_clusters: An integer number representing the (maximum) number
            of clusters to generate
        mode (default: "kmeans"): the clustering algorithm to apply. Can be
            'kmeans' or 'spc' or 'umap'
        transform (optional): function that maps waveforms to a new
            feature space
        min_cluster_size: Integer indicating minimum cluster size. Clusters smaller
            than this value will be assigned the label -1.

    Returns:
        Numpy integer array representing labels for each cluster found
    """
    _fn_start = time.time()
    _new_labels = -1 * np.ones(len(dataset)).astype(np.int)
    # len_last_window = dpoints

    if not len(dataset):
        return np.array([])

    for level in range(levels):
        # At each level of the hierarchical clustering, take only points
        # that haven't been clustered by a lower level yet
        remaining_data = dataset.select(_new_labels == -1)
        remaining_labels = _new_labels[_new_labels == -1]
        for i in range(0, len(remaining_data), dpoints):
            # Indexes relative to remaining_data
            next_window = np.arange(i, min(i + dpoints, len(remaining_data)))

            if len(next_window) < n_components:
                # Not enough points left over
                break

            # if len(next_window) < dpoints and len(next_window) == len_last_window:
            #     # The last level we 
            #     break
            # len_last_window = len(next_window)

            window_data = remaining_data.select(next_window)

            if remaining_data.has_children:
                weights = np.array([node.count for node in window_data.nodes])
            else:
                weights = None

            decomp = PCA(n_components=6).fit_transform(window_data.waveforms)
            if mode == "kmeans":
                clusterer = KMeans(n_clusters=n_components)
                clusterer.fit(decomp, sample_weight=weights)
                labels = clusterer.predict(decomp, sample_weight=weights)
                neighbor_cleaner = KNeighborsClassifier(n_neighbors=10).fit(decomp, labels)
                labels = neighbor_cleaner.predict(decomp)
            elif mode == "spc":
                clusterer = SPC(n_neighbors=10)
                clusterer.fit(decomp)
                result = clusterer.create_hierarchy()
                result = clusterer.collapse(result, threshold=10.0)
                labels = result.labels()
            elif mode == "umap":
                clusterer = KMeans(n_clusters=n_components)
                decomp = umap.UMAP(n_components=6).fit_transform(window_data.waveforms)
                clusterer.fit(decomp, sample_weight=weights)
                labels = clusterer.predict(decomp, sample_weight=weights)
                neighbor_cleaner = KNeighborsClassifier(n_neighbors=10).fit(decomp, labels)
                labels = neighbor_cleaner.predict(decomp)



            for label, count in zip(*np.unique(labels, return_counts=True)):
                # At the last level, only give isolated datapoints the -1 label
                # if level == levels - 1 and count == 1:
                #     labels[labels == label] = -1
                if count < min_cluster_size:
                    labels[labels == label] = -1
                else:
                    labels[labels == label] += np.max(
                        list(set(_new_labels) | set(remaining_labels))
                    ) + 1

            remaining_labels[next_window] = labels
            print(
                "Completed {}/{} in {:.1f}s.".format(
                    np.max(next_window),
                    len(remaining_data), time.time() - _fn_start
                ),
                end="\r")
        _new_labels[_new_labels == -1] = remaining_labels

    print("Completed clustering in {:.1f}s\n".format(time.time() - _fn_start))
    return _new_labels


def compute_peak(node):
    node = node.flatten()
    return np.mean(node.waveforms, axis=0)[node.waveforms.shape[1] // 2]


def compute_snr(node):
    node = node.flatten()
    std = np.std(node.waveforms, axis=0)
    mean = np.mean(node.waveforms, axis=0)
    snr = (np.max(mean) - np.min(mean)) / np.mean(std)
    return snr


def sort(dataset, resume_from=None):
    if resume_from is None:
        resume_from = []

    if len(resume_from) != 0:
        clustered = resume_from[0]
    else:
        print("Clustering {}".format(dataset))
        split = SplitDataset(dataset, (dataset.waveforms[:, dataset.waveforms.shape[1] // 2]) > 20)

        upgoing_labels = cluster_step(
            split.set_1,
            dpoints=1000,
            min_cluster_size=10,
            n_components=50,
            levels=5,
            mode="kmeans"
        )
        downgoing_labels = cluster_step(
            split.set_2,
            dpoints=1000,
            min_cluster_size=10,
            n_components=50,
            levels=5,
            mode="kmeans"
        )

        split.cluster(upgoing_labels, downgoing_labels)

        upgoing_labels = cluster_step(
            split.set_1,
            dpoints=500,
            min_cluster_size=5,
            n_components=50,
            levels=5,
            mode="umap"
        )
        downgoing_labels = cluster_step(
            split.set_2,
            dpoints=500,
            min_cluster_size=5,
            n_components=50,
            levels=5,
            mode="umap"
        )

        split.cluster(upgoing_labels, downgoing_labels)

        clustered = split.recombine()

        print("After two rounds of clustering:\n{}".format(clustered))
        yield clustered

    if len(resume_from) > 1:
        clustered = resume_from[1]
    else:
        split_2 = SplitDataset(clustered, (clustered.waveforms[:, clustered.waveforms.shape[1] // 2]) > 0)

        if len(split_2.set_1) > 10:
            upgoing_labels = spc_clustering(split_2.set_1)
            umapped_upgoing = umap.UMAP(n_components=3).fit_transform(split_2.set_1.waveforms)
            knn = KNeighborsClassifier(n_neighbors=10).fit(umapped_upgoing, upgoing_labels)
            upgoing_labels = knn.predict(umapped_upgoing)
        else:
            upgoing_labels = split_2.set_1.labels

        if len(split_2.set_2) > 10:
            downgoing_labels = spc_clustering(split_2.set_2)
            umapped_downgoing = umap.UMAP(n_components=3).fit_transform(split_2.set_2.waveforms)
            knn = KNeighborsClassifier(n_neighbors=10).fit(umapped_downgoing, downgoing_labels)
            downgoing_labels = knn.predict(umapped_downgoing)
        else:
            downgoing_labels = split_2.set_2.labels

        split_2.cluster(upgoing_labels, downgoing_labels)
        clustered = split_2.recombine()

        print("Combining clusters:\n{}".format(clustered))
        yield clustered

    if len(resume_from) > 2:
        clustered = resume_from[2]
    else:
        peaks = np.array([compute_peak(node) for node in clustered.nodes])
        snrs = np.array([compute_snr(node) for node in clustered.nodes])
        peak_value = np.abs(peaks[np.argmax(snrs)]) * 2  # Exclude clusters with peaks larger than twice the max snr peak

        clustered = clustered.select(np.abs(peaks) < peak_value)

        skews = dict(
            (label, compute_skew(node.flatten().waveforms[:, node.waveforms.shape[1] // 2]))
            for label, node in clustered.labeled_nodes
        )
        skews = np.array([skews[l] for l in clustered.flatten(1).labels])

        split_3 = SplitDataset(clustered.flatten(1), skews > -1.0)

        # Combine low skewed (high snr) clusters
        low_skew_labels = eliminate_small_clusters(
            split_3.set_1,
            split_3.set_1.labels,
            mode="high_snr"
        )

        # Cluster highly skewed clusters aggresively since then are smaller and higher snr
        # And weight time more highly
        if len(split_3.set_2):
            high_skew_labels = hdb_clustering(
                split_3.set_2,
                min_cluster_size=20,
                repeat=5,
                real_min_cluster_size=10000
            )
        else:
            high_skew_labels = np.array([])

        split_3.cluster(low_skew_labels, high_skew_labels)
        clustered = split_3.recombine()

        '''
        clustered = clustered.flatten()

        split_3 = SplitDataset(clustered, (clustered.waveforms[:, clustered.waveforms.shape[1] // 2]) > 0)
        upgoing_labels = relabel(split_3.set_1, n_components=3)  # Just because there tends to be fewer upgoing
        downgoing_labels = relabel(split_3.set_2, n_components=5)
        split_3.cluster(upgoing_labels, downgoing_labels)
        clustered = split_3.recombine()
        '''

        print("Reclustered:\n{}".format(clustered))
        yield clustered

    '''
    if len(resume_from) == 4:
        raise Exception("Already ran all possible")
    else:
        split_4 = SplitDataset(clustered, (clustered.waveforms[:, clustered.waveforms.shape[1] // 2]) > 0)

        upgoing_labels = spc_clustering(split_4.set_1)
        if len(upgoing_labels):
            umapped_upgoing = umap.UMAP(n_components=3).fit_transform(split_4.set_1.waveforms)
            knn = KNeighborsClassifier(n_neighbors=10).fit(umapped_upgoing, upgoing_labels)
            upgoing_labels = knn.predict(umapped_upgoing)

        downgoing_labels = spc_clustering(split_4.set_2)
        if len(downgoing_labels):
            umapped_downgoing = umap.UMAP(n_components=3).fit_transform(split_4.set_2.waveforms)
            knn = KNeighborsClassifier(n_neighbors=10).fit(umapped_downgoing, downgoing_labels)
            downgoing_labels = knn.predict(umapped_downgoing)

        split_4.cluster(upgoing_labels, downgoing_labels)

        clustered = split_4.recombine()

        print("Finished:\n{}".format(clustered))
        yield clustered
    '''
