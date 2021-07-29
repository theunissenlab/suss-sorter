import time

# import hdbscan
import networkx as nx
import numpy as np
import scipy.stats
from scipy import integrate
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, kneighbors_graph
from sklearn.mixture import BayesianGaussianMixture

from sklearn.manifold import TSNE

from .core import SpikeDataset


def threshold_graph(g, threshold):
    graph = nx.Graph()
    for i, j in g.edges:
        weight = g[i][j]['weight']
        if weight >= threshold or weight == np.max([g[i][k]["weight"] for k in g[i]]):
            graph.add_edge(i, j, weight=weight)
    return graph

def get_weights(graph):
    return [graph[i][j]["weight"] for i, j in graph.edges]


def get_mknn(X, n_neighbors=10):
    _dist = scipy.spatial.distance_matrix(X, X)
    spanning = minimum_spanning_tree(_dist).toarray()
    knn = kneighbors_graph(X, n_neighbors=n_neighbors, mode="distance").toarray()
    knn = knn * (knn.T > 0)
    knn[knn == 0] = spanning[knn == 0]
    return nx.from_numpy_array(knn)


def remove_outliers(mknn, n_neighbors=10, edges=1):    
    degs = [n for n, d in mknn.degree() if d < edges + 1]
    while len(degs):
        mknn.remove_nodes_from(degs)
        degs = [n for n, d in mknn.degree() if d < edges + 1]

    return mknn


def label_outliers(X, n_neighbors=10):
    mknn = get_mknn(X)
    mknn = remove_outliers(mknn, n_neighbors=n_neighbors, edges=2)
    labels = np.ones(len(X))
    labels[np.array(mknn)] = 0
    return mknn, labels


class Node(object):
    def __init__(self, ids, level=0, parent=None):
        self.ids = ids
        self.parent = parent
        self.level = level
        self.children = []
        
    def add_ids(self, ids):
        self.ids = np.concatenate([self.ids, ids])
    
    def add_child(self, child):
        self.children.append(child)
        
    def leaves(self, level=None):
        result = []
        if not self.children:
            return [self]
        
        if level and (
                self.level > level or
                self.children[0].level > level):
            return [self]
        
        for child in self.children:
            result += child.leaves(level=level)
        
        return sorted(result, reverse=True, key=lambda x: x.level)
    
    def leaf_parents(self):
        result = []
        
        if not self.children:
            return []
    
        if np.all([len(child.children) == 0 for child in self.children]):
            return [self]
        else:
            for child in self.children:
                result += child.leaf_parents()
        
        return sorted(result, reverse=True, key=lambda x: x.level)
    
    def __repr__(self):
        return "Level {} tree (n={}) with {} children".format(self.level, len(self), len(self.children))

    def __len__(self):
        return len(self.ids)
    
    def cluster(self, labels, level=None):
        unique = np.unique(labels)
        for label in unique:
            new_node = Node(
                self.ids[labels == label],
                level=level or self.level + 1,
                parent=self)
            self.add_child(new_node)

    def labels(self, level=None):
        labels = -1 * np.ones_like(self.ids)
        for i, leaf in enumerate(self.leaves(level=level)):
            labels[leaf.ids] = i
        return labels
        

class SPC(object):
    
    def __init__(self, n_neighbors=10, p_thresh=0.5):
        self.p_thresh = p_thresh
        self.n_neighbors = n_neighbors
        
    def fit(self, data):
        self._dist = scipy.spatial.distance_matrix(data, data)
        self.data = data
        
        spanning = minimum_spanning_tree(self._dist).toarray()
        knn = kneighbors_graph(data, n_neighbors=self.n_neighbors, mode="distance").toarray()
        knn = knn * (knn.T > 0)
        knn[knn == 0] = spanning[knn == 0]
        self._graph = nx.from_numpy_array(knn)
        
        self._interaction = self.compute_interaction_graph()
            
    def compute_interaction_graph(self):
        avg_neighbors = 2 * len(self._graph.edges) / len(self._graph)
        avg_distance = np.mean([self._graph[i][j]["weight"] for i, j in self._graph.edges])
        
        J = (1 / avg_neighbors) * np.exp(-self._dist / (2 * avg_distance))
        
        graph = nx.Graph()        
        for i, j in self._graph.edges:
            graph.add_edge(i, j, weight=1000 * (-J[i][j] / np.log(1 - self.p_thresh)) ** 3)
        return graph
    
    def predict(self, T):
        thresholded = threshold_graph(self._interaction, T)
        labels = -1 * np.ones(len(self.data))
        for label, idxs in enumerate(nx.connected_components(thresholded)):
            labels[np.array(self._interaction.subgraph(idxs))] = label
        return labels
    
    def max_temp(self):
        return np.max(get_weights(self._interaction))
    
    def max_cluster_size(self, labels):
        label_ids, label_counts = np.unique(labels, return_counts=True)
        return np.max(label_counts[label_ids != -1])

    def find_temp(self, min_cluster_size=None):
        if min_cluster_size is None:
            min_cluster_size = len(self.data) / 100.0
            
        t_low = 0.0
        t_high = self.max_temp()
        min_step = t_high / 100.0
        labels = self.predict(T=t_high)
        curr_max_cluster = self.max_cluster_size(labels)
        while (t_high - t_low) > min_step:
            if curr_max_cluster > min_cluster_size:
                t_low = t_high
                t_high = 2 * t_high
                labels = self.predict(T=t_high)
                curr_max_cluster = self.max_cluster_size(labels)
                continue
            t = (t_high + t_low) / 2
            labels = self.predict(T=t)
            max_cluster = self.max_cluster_size(labels)
            if max_cluster < min_cluster_size:
                t_high = t
                curr_max_cluster = max_cluster
            else:
                t_low = t
                
        return t_high
    
    def create_hierarchy(self, t_max=None, min_cluster_size=None):
        if t_max is None:
            t_max = self.max_temp()
            
        if min_cluster_size is None:
            min_cluster_size = len(self.data) / 100.0
        
        root = Node(np.arange(len(self.data)))

        for t_idx, t in enumerate(np.linspace(0, t_max, 50)):
            labels = self.predict(T=t)
            for node in root.leaves():
                node_labels = labels[node.ids]
                cluster_labels, cluster_sizes = np.unique(node_labels, return_counts=True)
                
                cluster_labels = np.unique(cluster_labels[
                    (cluster_sizes > min_cluster_size) &
                    (cluster_labels != -1) &
                    ((len(node) - cluster_sizes) > min_cluster_size)
                ])
                node_labels[np.logical_not(np.isin(node_labels, cluster_labels))] = -1
#                 if len(np.unique(node_labels)) == 1:
#                     continue
                node.cluster(node_labels, level=t_idx)

        return root
    
    def collapse(self, tree, threshold=50.0):
        while len(tree.leaf_parents()) > 1:
            to_update = []
            for node in tree.leaf_parents():
                for child in node.children:
                    child.iso = compute_isolation(self.data, child, node)

                isolated = [child for child in node.children if child.iso > threshold]
                bad_nodes = [child for child in node.children if child.iso <= threshold]
                last_node = None
                while bad_nodes:
                    bad_node = bad_nodes.pop()
                    for isolated_node in isolated:
                        if compute_isolation(self.data, bad_node, isolated_node) <= threshold:
                            isolated_node.add_ids(bad_node.ids)
                            break
                    else:
                        if last_node:
                            last_node.add_ids(bad_node.ids)
                        else:
                            last_node = bad_node

                if last_node is not None:
                    isolated.append(last_node)

                node.children = isolated
                if node.parent not in to_update:
                    to_update.append(node.parent)

            for parent in to_update:
                parent.children = [childs_child for child in parent.children for childs_child in child.children]
                for child in parent.children:
                    child.level -= 1

        return tree


def compute_isolation(data, leaf, parent):
    l1 = data[leaf.ids]
    l2 = data[parent.ids]
    
    l = np.mean(l1, axis=0)
    p = np.mean(l2, axis=0)
    dl = np.sum((l1 - l) ** 2)
    dp = np.sum((l1 - p) ** 2)
    if dl >= dp:
        return 0
    return np.sqrt(dp - dl) / data.shape[1]



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
    # while -1 in _new_labels:

    for direction in ["upgoing", "downgoing"]:
        if direction == "upgoing":
            remaining_data = dataset.select(dataset.waveforms[:, dataset.waveforms.shape[1] // 2] > 0)
            print("Upgoing, len={}".format(len(remaining_data)))
        else:
            remaining_data = dataset.select(dataset.waveforms[:, dataset.waveforms.shape[1] // 2] < 0)
            print("Downgoing, len={}".format(len(remaining_data)))

        for _ in range(4):
            remaining_data = remaining_data.select(_new_labels[remaining_data.ids] == -1)
            print("\nRound {}\n".format(_))
            for i in range(0, len(remaining_data), dpoints):
                # next_window = np.where(_new_labels == -1)[0][:dpoints]
                next_window = np.arange(i, min(i + dpoints, len(remaining_data)))

                if len(next_window) < n_components:
                    break

                if len(next_window) < dpoints and len(next_window) == len_last_window:
                    break

                len_last_window = len(next_window)

                window_data = remaining_data.select(next_window)
                if remaining_data.has_children:
                    weights = np.array([node.count for node in window_data.nodes])
                else:
                    weights = None
                if mode == "kmeans":
                    clusterer = KMeans(n_clusters=n_components)
                    clusterer.fit(window_data.waveforms, sample_weight=weights)
                    labels = clusterer.predict(window_data.waveforms, sample_weight=weights)
                elif mode == "spc":
                    clusterer = SPC(n_neighbors=5)
                    tsned = PCA(n_components=6).fit_transform(window_data.waveforms)
                    clusterer.fit(tsned)
                    result = clusterer.create_hierarchy()
                    result = clusterer.collapse(result, threshold=30.0)
                    labels = result.labels()

                for label, count in zip(*np.unique(labels, return_counts=True)):
                    if count < min_cluster_size:
                        labels[labels == label] = -1
                    else:
                        labels[labels == label] += np.max(_new_labels) + 1

                _new_labels[remaining_data.ids[next_window]] = labels
                print(
                    "Completed {}/{} in {:.1f}s.".format(
                        np.max(next_window),
                        len(remaining_data), time.time() - _fn_start
                    ),
                    end="\r")
            print(_new_labels[remaining_data.ids][:20])

        # yield dataset.cluster(_new_labels)

    print("Completed clustering in {:.1f}s".format(time.time() - _fn_start))
    return dataset.cluster(_new_labels)


def reassign_unassigned(waveforms, labels):
    neigh = KNeighborsClassifier(n_neighbors=3)
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
            np.mean(flat.waveforms[flat.labels == label], axis=0)
        )
        for label in np.unique(flat.labels)
    )

    if len(flat.ids):
        dataset.waveforms[:] = [centroids[label] for label in flat.labels]

    return denoised_node


def denoising_sort(times, waveforms):
    spike_dataset = SpikeDataset(times=times, waveforms=waveforms)

    original_waveforms = spike_dataset.waveforms.copy()

    steps = [
        dict(min_waveforms=30, dpoints=1000, n_components=30, mode="kmeans"),
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


def point_quality(data, labels, n_neighbors=3):
    badness = np.zeros(len(labels))
    for label in np.unique(labels):
        cluster_size = len(np.where(labels == label)[0])
        if cluster_size == 1:
            continue

        take_n = min(n_neighbors, cluster_size)
        neighbors = NearestNeighbors(
            n_neighbors=take_n,
            algorithm="ball_tree"
        ).fit(data[labels == label])

        dist, indices = neighbors.kneighbors(data[labels == label])
        dists = dist[:, 1:take_n]

        badness[labels == label] = np.mean(dists, axis=1)

    return scipy.stats.zscore(badness)


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
    if len(data) <= n_neighbors:
        n_neighbors = 2
    cleaner = KNeighborsClassifier(n_neighbors=n_neighbors)
    cleaner.fit(data, labels)
    labels = cleaner.predict(data)
    return labels


def flip_points(data, labels, flippable, n_neighbors=10, create_labels=False):
    raise Exception("Currently not in use")

    if np.sum(flippable) == 0:
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


def tsne_time(dataset, perplexity=30, t_scale=2 * 60 * 60, pcs=12):
    if pcs >= min(dataset.waveforms.shape):
        pcaed = PCA(n_components=min(dataset.waveforms.shape)).fit_transform(dataset.waveforms)
    else:
        pcaed = PCA(n_components=pcs).fit_transform(dataset.waveforms)
    wf_arr = scipy.stats.zscore(pcaed)
    t_arr = dataset.times / t_scale
    t_arr = t_arr - np.mean(t_arr)

    return TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=5000,
            n_iter_without_progress=500
    ).fit_transform(np.hstack([wf_arr, t_arr[:, None]]))


def pca_time(dataset, t_scale=2 * 60 * 60, pcs=6):
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
    raise Exception("Currently not in use")
    temp_dataset = dataset
    final_labels = -1 * np.ones(len(dataset)).astype(np.int)
    cluster_size = 6
    for idx in range(n):
        mask = final_labels == -1

        if np.sum(mask) < 100:
            break

        temp_dataset = dataset.select(mask)
        tsned = tsne_time(temp_dataset)

        hdb = hdbscan.HDBSCAN(min_cluster_size=cluster_size)
        labels = hdb.fit_predict(tsned)
        if -1 in labels:
            labels = reassign_unassigned(tsned, labels)

        quality = cluster_quality(tsned, labels)
        isolated = is_isolated(labels, quality)
        if not len(isolated) and cluster_size > 2:
            cluster_size -= 1
        else:
            cluster_size += 1 
        labels[np.logical_not(isolated)] = -1
        labels[labels != -1] += np.max(final_labels) + 1
        final_labels[mask] = labels

    result = dataset.cluster(final_labels)
    return result
    return result.select(
            [isi(n) < 0.05 for n in result.nodes],
            child=False)


def denoise(times, waveforms):
    threshold = np.log(0.001)

    pcaed = PCA(n_components=2).fit_transform(waveforms)
    pcaed = scipy.stats.zscore(pcaed, axis=0)
    mix = BayesianGaussianMixture(n_components=2).fit(pcaed)
    logprob = mix.score_samples(pcaed)
    times = times[logprob > threshold]
    waveforms = waveforms[logprob > threshold]

    denoised = denoising_sort(times, waveforms)
    # denoised = denoised.select([isi(n) < 0.05 for n in denoised.nodes])

    denoised = cluster_step(
        denoised,
        dpoints=200,
        n_components=10,
        min_cluster_size=3,
        mode="kmeans"
    )
    # denoised = denoised.select([isi(n) < 0.05 for n in denoised.nodes])
    return denoised


def _vote_on_labels(dataset):
    tsned = tsne_time(dataset, pcs=6, t_scale=2 * 60 * 60)
    spc = SPC(n_neighbors=min(10, len(dataset) - 1))
    spc.fit(tsned)
    result = spc.create_hierarchy()
    result = spc.collapse(result, threshold=20.0)
    labels = result.labels()
    return cleanup_clusters(tsned, labels, n_neighbors=3)


def sort(denoised):
    denoised_pcaed = pca_time(denoised, t_scale=6 * 60 * 60, pcs=3)
    # denoised_pcaed = scipy.stats.zscore(denoised_pcaed, axis=0)

    _, outliers = label_outliers(denoised_pcaed, n_neighbors=2)

    denoised = denoised.select(outliers == 0)
    denoised_pcaed = pca_time(denoised, t_scale=6 * 60 * 60, pcs=3)

    labels = []
    for _ in range(4):
        labels.append(_vote_on_labels(denoised))

    label_map = {}
    next_label = 0
    final_labels = np.zeros_like(labels[0])
    for idx, label_key in enumerate(zip(*labels)):
        key = tuple(label_key)
        if key not in label_map:
            label_map[key] = next_label
            next_label += 1
        final_labels[idx] = label_map[key]

    for label, count in zip(*np.unique(final_labels, return_counts=True)):
        if count == 1:
            final_labels[final_labels == label] = -1

    final_labels = cleanup_clusters(denoised_pcaed, final_labels, n_neighbors=5)
    final_labels = cleanup_clusters(denoised_pcaed, final_labels, n_neighbors=5)
    return denoised.cluster(final_labels)
