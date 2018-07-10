from collections import defaultdict

import numpy as np
from PyQt5 import QtWidgets as widgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.decomposition import PCA


class TimeseriesPlot(widgets.QFrame):

    ndim = 1

    def __init__(self, size=(700, 100), parent=None):
        super().__init__(parent)
        self.size = size

        self._rotation_period = 800
        self.setup_plots()
        self.setup_data()
        self.init_ui()
        self.parent().UPDATED_CLUSTERS.connect(self.reset)
        self.parent().CLUSTER_HIGHLIGHT.connect(self.on_cluster_highlight)
        self.parent().CLUSTER_SELECT.connect(self.on_cluster_select)

    def reset(self):
        # Delete the old layout
        # widgets.QWidget().setLayout(self.layout())
        for label, scatters in self.scatters.items():
            for scat in scatters:
                scat.remove()
            del scatters[:]
        for scat in self.main_scatters:
            scat.remove()
        del self.main_scatters[:]
        self.setup_data()

    @property
    def dataset(self):
        return self.parent().dataset

    @property
    def colors(self):
        return self.parent().colors

    @property
    def selected(self):
        return self.parent().selected

    def setup_plots(self):
        fig = Figure()
        fig.patch.set_alpha(0.0)
        self.canvas = FigureCanvas(fig)
        self.canvas.setStyleSheet("background-color:transparent;")

        self.axes = []
        for dim in range(self.ndim):
            ax = fig.add_axes(
                [0, 0.1 + (0.9 / self.ndim) * dim, 1, 0.9 / self.ndim],
                facecolor="#111111"
            )
            ax.set_yticks([])
            ax.patch.set_alpha(0.8)
            self.axes.append(ax)

        for ax in self.axes[1:]:
            ax.set_yticks([])
            ax.set_xticks([])

        self.scatters = defaultdict(list)
        self.main_scatters = []

        self._frame = 0
        self.parent().animation_timer.timeout.connect(self.rotate)

    def rotate(self):
        if not len(self.dataset.nodes) or not self.main_scatters:
            return

        _t = (
                (2 * np.pi) *
                (self._frame % self._rotation_period) /
                self._rotation_period
        )
        for dim in range(self.ndim):
            self.main_scatters[dim].set_offsets(np.array([
                self.flattened.times,
                (
                    np.cos(_t + dim * np.pi / 2) * self.pcs.T[dim] +
                    np.sin(_t + dim * np.pi / 2) * self.pcs.T[dim + 1]
                )
            ]).T)
        for label, node in zip(self.dataset.labels, self.dataset.nodes):
            pcs = self.pcs[self.flattened.labels == label]
            for dim in range(self.ndim):
                self.scatters[label][dim].set_offsets(np.array([
                    self.flattened.times[self.flattened.labels == label],
                    (
                        np.cos(_t + dim * np.pi / 2) * pcs.T[dim] +
                        np.sin(_t + dim * np.pi / 2) * pcs.T[dim + 1]
                    )
                ]).T)
        self._frame += 1
        self._frame = self._frame % self._rotation_period
        self.canvas.draw_idle()

    def setup_data(self):
        if not len(self.dataset.nodes):
            self.canvas.draw_idle()
            return

        self.scatters = defaultdict(list)
        self.main_scatters = []

        self.flattened = self.dataset.flatten(1)

        self.pca = PCA(n_components=self.ndim + 1).fit(self.flattened.waveforms)
        self.pcs = self.pca.transform(self.flattened.waveforms)

        for dim in range(self.ndim):
            self.main_scatters.append(self.axes[dim].scatter(
                self.flattened.times,
                self.pcs.T[dim],
                s=5,
                alpha=0.8,
                color="Gray",
                rasterized=True
            ))

        for label, node in zip(self.dataset.labels, self.dataset.nodes):
            pcs = self.pcs[self.flattened.labels == label]
            for dim in range(self.ndim):
                self.scatters[label].append(
                    self.axes[dim].scatter(
                        node.times,
                        pcs.T[dim],
                        s=10,
                        alpha=1,
                        color=self.colors[label],
                        rasterized=True
                    )
                )
                self.scatters[label][-1].set_visible(False)

        for dim, ax in enumerate(self.axes):
            ylim = np.max(np.abs(self.pcs[:, dim])) * 1.2
            ax.set_ylim(-ylim, ylim)
            fully_flat = self.dataset.flatten().times
            xlim = (fully_flat[0], fully_flat[-1])
            ax.set_xlim(*xlim)

        self.canvas.draw_idle()

    def on_cluster_select(self, selected, old_selected):
        for label in self.scatters:
            for scat in self.scatters[label]:
                # TODO (kevin): i dont think this set color is needed
                # scat.set_color(self.colors[label])
                scat.set_visible(label in selected)

    def on_cluster_highlight(self, new_highlight, old_highlight, temporary):
        if (
                old_highlight is not None and
                old_highlight in self.scatters and
                old_highlight in self.colors
                ):
            for scat in self.scatters[old_highlight]:
                scat.set_color(self.colors[old_highlight])
                scat.set_visible(old_highlight in self.selected)

        if new_highlight is not None:
            for scat in self.scatters[new_highlight]:
                scat.set_facecolor("White")
                scat.set_edgecolor(self.colors[new_highlight])
                scat.set_visible(True)

    def init_ui(self):
        layout = widgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
