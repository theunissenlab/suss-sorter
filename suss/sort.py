import warnings

import networkx as nx
import numpy as np
import scipy
import scipy.stats
import umap
from isosplit5 import isosplit5
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.neighbors import KNeighborsClassifier

from suss.core import SpikeDataset


ICA_COMPONENTS = 12
GMM_COMPONENTS = 6
AREA_GAP_THRESHOLD = 0.05
DENSITY_RATIO_THRESHOLD = 2
KS_THRESHOLD = 0.5


def cleanup_clusters(data, labels, n_neighbors=20):
    if len(data) <= n_neighbors:
        n_neighbors = 2
    cleaner = KNeighborsClassifier(n_neighbors=n_neighbors)
    cleaner.fit(data, labels)
    labels = cleaner.predict(data)
    return labels


def pca_time(dataset, t_scale=2 * 60 * 60, pcs=6):
    pcaed = PCA(n_components=pcs).fit_transform(dataset.waveforms)
    wf_arr = scipy.stats.zscore(pcaed, axis=0)
    t_arr = dataset.times / t_scale
    t_arr = t_arr - np.mean(t_arr)

    return PCA(n_components=pcs).fit_transform(
        np.hstack([wf_arr, t_arr[:, None]])
    )


def lda_2d(wfs, labels, l1, l2):
    """Project waveforms into space discriminting l1 (or l1 and l2)
    
    Return the data divided into l1, l2 and other
    """
    labels_ = np.zeros_like(labels)
    labels_[labels == l1] = 1
    labels_[labels == l2] = 2
    
    if len(np.unique(labels_)) > 2:
        lda = LDA(n_components=3).fit(wfs, labels_)
    else:
        lda = PCA(n_components=2).fit(wfs)

    wfs1 = wfs[labels == l1]
    wfs2 = wfs[labels == l2]
    wfs_ = wfs[np.logical_not(np.isin(labels, [l1, l2]))]

    return (
        lda.transform(wfs1),
        lda.transform(wfs2),
        np.zeros((0, 1)) if not len(wfs_) else lda.transform(wfs_)
    )


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


def guided_reassignment(X, y=None, force_clusters=None):
    """Reassign datapoints into new clusters
    
    Args
    ----
    X: data array
    y: initial labels
    force_clusters: number of clusters to come up with
    """
    if force_clusters is not None:
        k = force_clusters - 1
    elif y is not None:
        k = len(np.unique(y)) - 1
    else:
        k = 2

    if y is None:
        y = GaussianMixture(n_components=k, max_iter=1000).fit_predict(X)
    
    projection = LDA(n_components=1).fit(X, y).transform(X)
    new_y = isosplit5(projection.T) - 1

    if force_clusters is not None and len(np.unique(new_y)) != force_clusters:
        return y
    
    return new_y


def split_window(wfs, labels, umapped):

    # Split clusters using densities and areas
    for label in np.arange(len(np.unique(labels))):

        if label == -1 or np.sum(labels == label) < 20:
            continue
        selector = np.where(labels == label)[0]
        sub_wfs = umapped[labels == label]  # (wfs[labels == label])
        sub_labels = guided_reassignment(sub_wfs)

        if len(np.unique(sub_labels)) > 1:
            for new_label in np.unique(sub_labels):
                if new_label > 0:
                    selector_ = np.zeros_like(labels)
                    for i in selector[sub_labels == new_label]:
                        selector_[i] = 1
                    np.place(labels, selector_.astype(bool), np.max(labels) + 1)

    final_labels = np.copy(labels)

    return final_labels


def cluster_window(wfs, projected):
    projection = FastICA(n_components=ICA_COMPONENTS, max_iter=1000).fit_transform(wfs)
    labels = BayesianGaussianMixture(n_components=GMM_COMPONENTS, max_iter=1000).fit_predict(projection)

    labels = split_window(wfs, labels, projected)

    return labels


def graph_to_labels(label_windows, G):
    total_labels = -1 * np.ones(np.sum([len(win) for win in label_windows]))
    
    step = 2000
    clusts = 6

    label_i = 0
    subgraphs = [G.subgraph(c) for c in nx.weakly_connected_components(G)]
    for subgraph in subgraphs:
        if len(subgraph.nodes) <= 2:
            continue
        for window_idx, cluster_label in subgraph.nodes:
            np.place(
                total_labels[window_idx * step:(window_idx + 1) * step],
                label_windows[window_idx] == cluster_label,
                label_i)
        label_i += 1
    
    return total_labels


def connect_windows(windows, label_windows, reassign=True):
    G = nx.DiGraph()

    for i in range(len(windows) - 1):
        # Compare window i and i + 1
        print("Comparing window {}/{}".format(i, len(windows)), end="\r")

        # Create links step
        for c1 in np.unique(label_windows[i]):
            for c2 in np.unique(label_windows[i + 1]):
                if c1 == -1 or c2 == -1:
                    continue

                pts1 = windows[i][1][label_windows[i] == c1]
                pts2 = windows[i + 1][1][label_windows[i + 1] == c2]
                if len(pts1) < 10 or len(pts2) < 10:
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    temp_pts1 = pts1[LocalOutlierFactor(n_neighbors=10, contamination=0.1).fit_predict(pts1) == True]
                    temp_pts2 = pts2[LocalOutlierFactor(n_neighbors=10, contamination=0.1).fit_predict(pts2) == True]
                    if len(temp_pts1) >= 5:
                        pts1 = temp_pts1
                    if len(temp_pts2) >= 5:
                        pts2 = temp_pts2

                if len(pts1) < 5 or len(pts2) < 5:
                    continue

                all_pts = np.concatenate([pts1, pts2])
                labels_ = np.concatenate([
                    np.zeros(len(pts1)),
                    np.ones(len(pts2))
                ])
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lda = LDA(n_components=1).fit(
                        all_pts, labels_
                    )

                k, p = scipy.stats.ks_2samp(lda.transform(pts1).flatten(), lda.transform(pts2).flatten())
                
                if 1 - k > 0.5:
                    G.add_edge((i, c1), (i + 1, c2))

        if reassign:
            # Reassignment Step
            # (Splits up nodes that have multiple nodes coming into it)
            for node in [n for n in G.nodes if n[0] == i + 1]:
                in_edges = G.in_edges(node)
                n_in_edges = len(in_edges)
                if n_in_edges > 1:
                    labels = [edge[0][1] for edge in in_edges]
                    prev_node = node[0] - 1
                    curr_node = node[0]
                    selector = np.where(label_windows[curr_node] == node[1])[0]
                    discriminator = LDA(n_components=n_in_edges - 1).fit(
                        windows[prev_node][1][np.isin(label_windows[prev_node], labels)],
                        label_windows[prev_node][np.isin(label_windows[prev_node], labels)]
                    )
                    X = windows[curr_node][1][selector]

                    new_labels = guided_reassignment(X, discriminator.predict(X), force_clusters=n_in_edges)

                    for label in np.unique(new_labels):
                        if label == 0:
                            continue
                        selector_ = np.zeros_like(label_windows[curr_node])
                        for k in selector[new_labels == label]:
                            selector_[k] = 1
                        np.place(label_windows[curr_node], selector_, np.max(label_windows[curr_node]) + 1)

    return label_windows, graph_to_labels(label_windows, G), G if not reassign else None


def cluster_leftovers(times, waveforms):
    """Reassign all datapoints labeled -1"""
    time_joined_waveforms = np.hstack([
        waveforms,
        (times - np.mean(times))[:, None] / 3600.0
    ])

    time_based_umap = umap.UMAP(n_components=3, n_neighbors=10, min_dist=0.01).fit_transform(
        time_joined_waveforms
    )

    label_groups = []
    step = 2000
    clusts = 6

    windows = [
        (None, waveforms[i:i+step])
        for i in range(0, len(waveforms), step)
    ]
    window_umapped = [
        time_based_umap[i:i+step]
        for i in range(0, len(waveforms), step)
    ]

    for i, (_, win) in enumerate(windows):
        print("{}/{}".format(i, len(windows)))
        label_groups.append(cluster_window(win, window_umapped[i]))

    label_groups, new_labels, G = connect_windows(windows, label_groups)
    label_groups, new_labels, G = connect_windows(windows, label_groups, reassign=False)

    return new_labels


def sort(times, waveforms):
    selector = LocalOutlierFactor(n_neighbors=20, contamination="auto").fit_predict(
        PCA(n_components=2).fit_transform(waveforms)
    ) == True
    waveforms_to_fit = waveforms[selector]
    waveforms_to_fit = waveforms_to_fit[::int(np.ceil(len(waveforms_to_fit) / 100000))]
    
    full_umap = umap.UMAP(
        n_components=3, n_neighbors=4, min_dist=0.1
    ).fit(waveforms_to_fit)

    step = 2000
    clusts = 6

    windows = [
        (times[i:i+step], waveforms[i:i+step])
        for i in range(0, len(times), step)
    ]

    label_groups = []
    for i, (_, win) in enumerate(windows):
        print("{}/{}".format(i, len(windows)), end="\r")
        label_groups.append(cluster_window(win, full_umap.transform(win)))

    label_groups, initial_labels, G = connect_windows(windows, label_groups)
    label_groups, initial_labels, G = connect_windows(windows, label_groups, reassign=False)

    new_labels = cluster_leftovers(
        times[initial_labels == -1],
        waveforms[initial_labels == -1],
    )

    updated_labels = np.copy(initial_labels)
    selector = np.where(initial_labels == -1)[0]
    for label in np.unique(new_labels):
        if label == -1:
            continue

        selector_ = np.zeros_like(initial_labels)
        for k in selector[new_labels == label]:
            selector_[k] = 1

        np.place(
            updated_labels,
            selector_,
            np.max(updated_labels) + 1
        )

    time_joined = np.hstack([
        PCA(n_components=20).fit_transform(waveforms[updated_labels == -1]),
        times[updated_labels == -1, None] / 3600.0
    ])
    leftover_labels = isosplit5(
        umap.UMAP(n_components=3, n_neighbors=10, min_dist=0.1).fit_transform(
            time_joined
        ).T
    )

    for l in np.unique(leftover_labels):
        sublabels = isosplit5(waveforms[updated_labels == -1][leftover_labels == l].T)
        select = np.where(leftover_labels == l)[0]
        np.put(
            leftover_labels,
            select,
            np.max(leftover_labels) + sublabels
        )

    final_labels = updated_labels.copy()
    np.place(
        final_labels,
        updated_labels == -1,
        leftover_labels + 1 + np.max(final_labels)
    )

    return times, waveforms, final_labels


def create_hierarchy(times, waveforms, labels, step=2000):
    windows = [
        (times[i:i+step], waveforms[i:i+step], labels[i:i+step])
        for i in range(0, len(waveforms), step)
    ]
    
    labels = []
    cluster_map = {}

    cluster_idx = 0
    label_idx = 0
    for i, (t, w, l) in enumerate(windows):
        label_map = {}
        for label in np.unique(l):
            label_map[label] = label_idx
            cluster_map[cluster_idx] = label
            cluster_idx += 1
            label_idx += 1

        for label in l:
            if label != -1:
                labels.append(label_map[label])
            else:
                labels.append(-1)
                
    dataset = SpikeDataset(times, waveforms)
    new_dataset = dataset.cluster(labels)
    final_dataset = new_dataset.cluster([cluster_map.get(l, -1) for l in new_dataset.labels])
    
    return final_dataset
