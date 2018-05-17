from collections import defaultdict
from functools import partial

import numpy as np
from PyQt5 import QtWidgets as widgets
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA


def get_color_dict(labels):
    unique_labels = sorted(np.unique(labels))
    n_labels = len(unique_labels)
    return dict((label, cm.gist_ncar(idx / n_labels))
            for idx, label in enumerate(unique_labels))


def selector_area(dataset, width, colors, cb):
    button_frame = widgets.QGroupBox()
    button_layout = widgets.QGridLayout()
    for label, cluster in zip(dataset.labels, dataset.nodes):
        button = widgets.QPushButton("{}".format(label))
        button.setCheckable(True)
        button.setDefault(False)
        button.setAutoDefault(False)
        button.clicked[bool].connect(partial(cb, label=label))
        color = "rgba({}, {}, {})".format(*(255 * np.array(colors.get(label))[:3]))
        button.setStyleSheet("""
            QPushButton:checked
            {{
                border: 4px solid #444444;
                border-style: inset;
            }}
            QPushButton {{
                background-color: {};
            }}
        """.format(color))
        button_layout.addWidget(button)
    button_frame.setLayout(button_layout)
    scroll_area = widgets.QScrollArea()
    scroll_area.setWidget(button_frame)
    scroll_area.setWidgetResizable(True)
    scroll_area.setFixedWidth(width)

    return scroll_area


def clear_axes(*axes):
    for ax in axes:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    return axes


class ProjectionsPane(widgets.QFrame):
    def __init__(self, dataset, colors,
            size=(300, 350),
            facecolor="#222222",
            frac=10,
            min_isi=0.001,
            parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.colors = colors
        self.size = size
        self.facecolor = facecolor
        self.frac = frac
        self.min_isi = min_isi
        self.active_clusters = set()

        self.setup_layout()
        self.setup_data()

    def setup_layout(self):
        fig = Figure()
        self.canvas = FigureCanvas(fig)
        # self.canvas.setFixedSize(*self.size)
        self.ax_1d = fig.add_axes([0, 0, 1, 0.2], facecolor=self.facecolor)
        self.ax_2d = fig.add_axes([0, 0.2, 1, 0.8], facecolor=self.facecolor)
        clear_axes(self.ax_1d, self.ax_2d)
        self.ax_2d.grid(color="#DDDDDD", linestyle="-", linewidth=0.2)

        layout = widgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def setup_data(self):
        pass

    def update_selection(self, active_clusters):
        self.active_clusters = active_clusters
        self.ax_1d.clear()
        self.ax_2d.clear()
        clear_axes(self.ax_1d, self.ax_2d)
        self.ax_2d.grid(color="#CCCCCC", linestyle="-", linewidth=0.2)

        if len(self.active_clusters) == 0:
            self.canvas.draw()
            return

        selected_clusters = self.dataset.select(
            np.isin(self.dataset.labels, list(self.active_clusters))
        )
        selected_flattened = selected_clusters.flatten(assign_labels=True)

        if len(self.active_clusters) == 1:
            lda2 = PCA(n_components=2).fit(selected_flattened.waveforms, selected_flattened.labels)
            lda1 = PCA(n_components=1).fit(selected_flattened.waveforms, selected_flattened.labels)
        elif len(self.active_clusters) == 2:
            lda2 = PCA(n_components=2).fit(selected_flattened.waveforms, selected_flattened.labels)
            lda1 = LDA(n_components=1).fit(selected_flattened.waveforms, selected_flattened.labels)
        else:
            lda2 = LDA(n_components=2).fit(selected_flattened.waveforms, selected_flattened.labels)
            lda1 = LDA(n_components=1).fit(selected_flattened.waveforms, selected_flattened.labels)

        # Transform data
        lda2_data = lda2.transform(selected_flattened.waveforms)
        lda1_data = lda1.transform(selected_flattened.waveforms)

        # Plot 1d projection as histogram
        labels = sorted(self.active_clusters)
        self.ax_1d.hist(
            [
                lda1_data[selected_flattened.labels == label].flatten()
                for label in labels
            ],
            bins=200,
            color=list(map(self.colors.get, labels)),
            alpha=0.9,
            stacked=True)

        # Plot 2d projections
        for label in self.active_clusters:
            self.ax_2d.scatter(
                *lda2_data[selected_flattened.labels == label][::self.frac].T,
                s=1,
                color=self.colors[label],
                alpha=1.0)

        # Draw violations on 2d scatter
        bad_idx = np.diff(selected_flattened.times) <= self.min_isi
        violations_within_cluster = bad_idx & (selected_flattened.labels[:-1] == selected_flattened.labels[1:])
        violations_across_cluster = bad_idx & (selected_flattened.labels[:-1] != selected_flattened.labels[1:])
        total_across = (selected_flattened.labels[:-1] != selected_flattened.labels[1:])

        within_pairs = zip(
            lda2_data[:-1][::self.frac][violations_within_cluster[::self.frac]],
            lda2_data[1:][::self.frac][violations_within_cluster[::self.frac]]
        )
        across_pairs = zip(
            lda2_data[:-1][::self.frac][violations_across_cluster[::self.frac]],
            lda2_data[1:][::self.frac][violations_across_cluster[::self.frac]]
        )

        for spike1, spike2 in within_pairs:
            self.ax_2d.plot(*zip(spike1, spike2), linewidth=0.8, color="#FFFFFF",
                    linestyle=":", alpha=1.0)

        for spike1, spike2 in across_pairs:
            self.ax_2d.plot(*zip(spike1, spike2), linewidth=1, color="#FFFFFF", alpha=1.0)
            self.ax_2d.plot(*zip(spike1, spike2), linewidth=0.5, color="Red",
                    linestyle="--", alpha=1.0)

        if len(self.active_clusters) >= 2:
            self.ax_2d.text(
                    np.max(self.ax_2d.get_xlim()),
                    np.max(self.ax_2d.get_ylim()),
                    "ISI across\n{:.1f}% (n={})".format(
                        100.0 * np.sum(violations_across_cluster) / np.sum(total_across),
                        int(np.sum(total_across))
                    ),
                    fontsize=10,
                    horizontalalignment="right",
                    verticalalignment="top",
            )

        self.canvas.draw()


class OverviewScatterPane(widgets.QFrame):

    def __init__(self, dataset, colors,
            size=(300, 300),
            inactive_color="#DDDDDD",
            facecolor="#FFFFFF",
            parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.colors = colors
        self.size = size

        self.inactive_color = inactive_color
        self.facecolor = facecolor

        self.setup_layout()
        self.setup_data()

    def setup_layout(self):
        fig = Figure()
        self.canvas = FigureCanvas(fig)
        self.canvas.setFixedSize(*self.size)
        self.ax = fig.add_axes([0, 0, 1, 1], facecolor=self.facecolor)
        clear_axes(self.ax)

        layout = widgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def setup_data(self):
        flattened = self.dataset.flatten(1, assign_labels=True)
        self.data = LDA(n_components=2).fit_transform(flattened.waveforms, flattened.labels)
        self.labels = flattened.labels

        for label in self.dataset.labels:
            self.ax.scatter(
                *self.data[self.labels == label].T,
                color=self.colors[label],
                alpha=0.1,
                s=2,
            )
        self.scatters = {}
        for label in self.dataset.labels:
            self.scatters[label] = self.ax.scatter(
                *self.data[self.labels == label].T,
                color=self.colors[label],
                alpha=1.0,
                s=1,
            )
            self.scatters[label].set_visible(False)
        self.canvas.draw()

    def update_selection(self, active_clusters):
        for label in self.scatters:
            self.scatters[label].set_visible(label in active_clusters)
        self.canvas.draw()


class LegendPane(widgets.QFrame):
    def __init__(self, selected_clusters, colors, parent=None):
        super().__init__(parent)


class ISIPane(widgets.QFrame):
    def __init__(self, dataset, colors, size=(250, 250), facecolor=None, frac=50, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.colors = colors
        self.size = size
        self.facecolor = facecolor
        self.frac = frac
        self.active_clusters = set()

        self.setup_layout()
        self.setup_data()

    def setup_layout(self):
        fig = Figure()
        self.canvas = FigureCanvas(fig)
        self.canvas.setFixedHeight(self.size[1])
        self.ax = fig.add_axes([0, 0, 1, 1], facecolor=self.facecolor)
        self.text_ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1), ylim=(0, 1))
        self.text_ax.patch.set_alpha(0.0)
        clear_axes(self.text_ax)

        layout = widgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def setup_data(self):
        pass

    def update_selection(self, active_clusters):
        self.active_clusters = active_clusters
        self.ax.clear()
        self.text_ax.clear()
        self.text_ax.patch.set_alpha(0.0)
        clear_axes(self.text_ax)

        if len(self.active_clusters) == 0:
            self.canvas.draw()
            return

        clusters = self.dataset.select(
            np.isin(self.dataset.labels, list(self.active_clusters))
        )

        flattened = clusters.flatten()

        isi = np.diff(flattened.times) 

        self.ax.hist(
            isi,
            bins=50,
            range=(0, 0.05),
            color="#888888",
            alpha=0.8,
            density=True
        )
        self.ax.vlines(0.001, *self.ax.get_ylim(), color="Red", linestyle="--")

        self.text_ax.text(0.5, 0.7,
                "{:.1f}%\nISI violations".format(
                    100.0 * np.sum(isi <= 0.001) / len(isi)
                ),
                fontsize=8,
                color="#BBBBBB"
        )

        self.canvas.draw()


class WaveformsPane(widgets.QFrame):
    def __init__(self, dataset, colors, size=(250, 250), facecolor=None, frac=50, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.colors = colors
        self.size = size
        self.facecolor = facecolor
        self.frac = frac
        self.active_clusters = set()

        self.setup_layout()
        self.setup_data()

    def setup_layout(self):
        fig = Figure()
        self.canvas = FigureCanvas(fig)
        # self.canvas.setFixedSize(*self.size)
        self.ax = fig.add_axes([0, 0, 1, 1], facecolor=self.facecolor,
                xlim=(0, 40),
                ylim=(-250, 120))
        clear_axes(self.ax)
        self.ax.grid(color="#CCCCCC", linestyle="-", linewidth=0.2)

        layout = widgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def setup_data(self):
        pass

    def update_selection(self, active_clusters):
        self.active_clusters = active_clusters
        self.ax.clear()
        clear_axes(self.ax)
        self.ax.set_xlim(0, 40)
        self.ax.grid(color="#CCCCCC", linestyle="-", linewidth=0.2)

        if len(self.active_clusters) == 0:
            self.canvas.draw()
            return

        selected_clusters = self.dataset.select(
            np.isin(self.dataset.labels, list(self.active_clusters))
        )
        flattened = selected_clusters.flatten(assign_labels=True)

        for label in self.active_clusters:
            line_segments = LineCollection(
                    [
                        np.column_stack([np.arange(len(wf)), wf])
                        for wf in flattened.waveforms[flattened.labels == label][::self.frac]
                    ],
                    color=self.colors[label],
                    linewidth=0.1,
                    alpha=0.7)
            self.ax.add_collection(line_segments)

        for label in self.active_clusters:
            self.ax.plot(
                np.mean(flattened.waveforms[flattened.labels == label], axis=0),
                linewidth=2.0,
                alpha=0.8,
                color="White"
            )
            self.ax.plot(
                np.mean(flattened.waveforms[flattened.labels == label], axis=0),
                linewidth=1.2,
                alpha=1.0,
                color=self.colors[label]
            )

        self.ax.plot(
                np.mean(flattened.waveforms, axis=0),
                linewidth=1,
                alpha=0.8,
                color="Black",
                linestyle="--"
        )
        self.canvas.draw()


class TimeseriesPane(widgets.QFrame):
    def __init__(self, dataset, colors,
            n_components=2,
            size=(700, 100),
            inactive_color="#DDDDDD",
            facecolor="#FFFFFF",
            frac=10,
            parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.colors = colors
        self.size = size
        self.frac = frac  # how many items to skip when plotting
        self.inactive_color = inactive_color
        self.facecolor = facecolor
        self.n_components=n_components
        self.scatters = {}

        self.setup_layout()
        self.setup_data()

    def setup_layout(self):
        fig = Figure()
        self.canvas = FigureCanvas(fig)
        # self.canvas.setFixedHeight(self.size[1])

        self.axes = []
        for component in range(self.n_components):
            self.axes.append(
                fig.add_axes(
                    [0, component / self.n_components, 1, 1 / self.n_components],
                    facecolor=self.facecolor
                )
            )
        clear_axes(*self.axes)

        layout = widgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def setup_data(self):
        for scatters in self.scatters.values():
            for scatter in scatters:
                scatter.remove()

        self.flattened = self.dataset.flatten(assign_labels=True)
        self.data = LDA(n_components=self.n_components).fit_transform(
            self.flattened.waveforms, self.flattened.labels)

        for component, ax in enumerate(self.axes):
            ax.scatter(
                self.flattened.times[::self.frac],
                self.data[::self.frac, component],
                s=2,
                alpha=0.1,
                color=self.inactive_color
            )
        self.scatters = defaultdict(list)
        for label in self.dataset.labels:
            for component, ax in enumerate(self.axes):
                self.scatters[label].append(ax.scatter(
                    self.flattened.times[self.flattened.labels == label][::self.frac],
                    self.data[self.flattened.labels == label][::self.frac, component],
                    s=2,
                    alpha=0.4,
                    color=self.colors[label]))
                self.scatters[label][-1].set_visible(False)

    def update_selection(self, active_clusters):
        for label in self.scatters:
            for scatter in self.scatters[label]:
                scatter.set_visible(label in active_clusters)
        self.canvas.draw()


class ClusterPane(widgets.QFrame):

    def __init__(self, cluster, color, size=(200, 50), parent=None):
        super().__init__(parent)
        self.cluster = cluster
        self.color = color
        self.size = size

        self.setup_layout()
        self.setup_data()

    def setup_layout(self):
        fig = Figure()
        self.canvas = FigureCanvas(fig)
        self.canvas.setFixedSize(*self.size)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.ax_wf = fig.add_axes([0, 0, 0.3, 1], facecolor="#BBBBBB")
        self.ax_isi = fig.add_axes([0.3, 0, 0.3, 1], facecolor="#CCCCCC")
        self.ax_skew = fig.add_axes([0.6, 0, 0.4, 1], facecolor="#BBBBBB")
        clear_axes(self.ax_wf, self.ax_isi, self.ax_skew)
        self.ax_skew.grid(color="#888888", linestyle=":", linewidth=1)
        self.ax_wf.grid(color="#888888", linestyle=":", linewidth=1)

        layout = widgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def setup_data(self):
        cluster = self.cluster.flatten()
        mean = np.mean(cluster.waveforms, axis=0)
        std = np.std(cluster.waveforms, axis=0)

        self.ax_wf.fill_between(np.arange(len(mean)), mean - std, mean + std, color=self.color, alpha=0.5)
        self.ax_wf.plot(mean, color=self.color, alpha=1.0, linewidth=2)
        self.ax_wf.vlines(0, 0, -50, linewidth=1, color="Black")

        isi = np.diff(cluster.times)
        isi_violations = np.sum(isi < 0.001) / len(isi)
        self.ax_isi.hist(isi, bins=50, range=(0, 0.05), density=True, color=self.color)
        # self.ax_isi.vlines(0.001, *self.ax_isi.get_ylim(), color="Red", linestyle="--")
        self.ax_isi.text(0.01, self.ax_isi.get_ylim()[1] * 0.7, "{:.1f}% < 1ms".format(100.0 * isi_violations), fontsize=5)

        peaks = np.min(cluster.waveforms, axis=1)
        self.ax_skew.hist(peaks, bins=100, density=True, color="#DDDDDD")
        skew = np.mean(((peaks - np.mean(peaks)) / np.std(peaks)) ** 3)
        self.ax_skew.text(np.mean(self.ax_skew.get_xlim()), self.ax_skew.get_ylim()[1] * 0.7, "skew: {:.1f}".format(skew), fontsize=5)
        self.canvas.draw()


class ClusterSelector(widgets.QScrollArea):
    """For viewing cluster data and hooks when clusters are selected
    """
    def __init__(self, dataset, colors, cb, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.colors = colors
        self.cb = cb

        self.setup_data()

    def setup_data(self):
        self.buttons = {}
        cluster_frame = widgets.QGroupBox()
        all_cluster_layout = widgets.QVBoxLayout()

        ordered_idx = np.argsort(self.dataset.labels)

        for label_idx in ordered_idx:
            label = self.dataset.labels[label_idx]
            cluster = self.dataset.nodes[label_idx]
            cluster_layout = widgets.QHBoxLayout()
            button = widgets.QPushButton("{} (n={})".format(label, cluster.waveform_count))
            button.setCheckable(True)
            button.setDefault(False)
            button.setAutoDefault(False)
            button.clicked[bool].connect(partial(self.cb, label=label))
            color = "rgba({}, {}, {})".format(*(255 * np.array(self.colors.get(label))[:3]))
            button.setStyleSheet("""
                QPushButton:checked
                {{
                    border: 4px solid #444444;
                    border-style: inset;
                }}
                QPushButton {{
                    background-color: {};
                }}
            """.format(color))
            self.buttons[label] = button
            cluster_layout.addWidget(button)
            cluster_layout.addWidget(ClusterPane(cluster, self.colors.get(label)))
            all_cluster_layout.addLayout(cluster_layout)
            
        cluster_frame.setLayout(all_cluster_layout)
        self.setWidget(cluster_frame)
        self.setWidgetResizable(True)
        self.setFixedWidth(380)

        # layout = widgets.QVBoxLayout()
        # layout.addWidget(cluster_frame)
        # self.setLayout(layout)

    def update_selection(self, active_clusters):
        for label in self.buttons:
            self.buttons[label].setChecked(label in active_clusters)


class ClusterManipulationOptions(widgets.QWidget):
    def __init__(self, reset_cb, clear_cb, merge_cb, save_cb, parent=None):
        super().__init__(parent)
        layout = widgets.QGridLayout(self)

        self.clear_button = widgets.QPushButton("Clear", self)
        self.reset_button = widgets.QPushButton("Reset", self)
        self.reset_button.setStyleSheet("QPushButton {background-color: red;}")
        self.merge_button = widgets.QPushButton("Merge", self)
        self.save_button = widgets.QPushButton("Save", self)
        self.save_button.setStyleSheet("QPushButton {background-color: green;}")
        self.reset_button.clicked.connect(reset_cb)
        self.clear_button.clicked.connect(clear_cb)
        self.merge_button.clicked.connect(merge_cb)
        self.save_button.clicked.connect(save_cb)

        layout.addWidget(self.reset_button, 0, 0, 1, 1)
        layout.addWidget(self.merge_button, 1, 0, 1, 1)
        layout.addWidget(self.save_button, 2, 0, 1, 1)

        self.setLayout(layout)
        self.setFixedWidth(100)
