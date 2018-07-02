import os
import sys
from collections import defaultdict
from functools import partial

import numpy as np
from PyQt5 import QtWidgets as widgets
from PyQt5.QtCore import Qt, QObject, QTimer, pyqtSignal
from PyQt5 import QtGui as gui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from suss.gui.utils import make_color_map, clear_axes, get_changed_labels


class WaveformsPlot(widgets.QFrame):

    def __init__(self, size=(700, 100), parent=None):
        super().__init__(parent)
        self._cached_cluster_stats = {}
        self.size = size

        self.setup_plots()
        self.setup_data()
        self.init_ui()
        self.parent().UPDATED_CLUSTERS.connect(self.reset)
        self.parent().CLUSTER_HIGHLIGHT.connect(self.on_cluster_highlight)
        self.parent().CLUSTER_SELECT.connect(self.on_cluster_select)

    def reset(self, new_dataset, old_dataset):
        for label in get_changed_labels(new_dataset, old_dataset):
            if label in self._cached_cluster_stats:
                del self._cached_cluster_stats[label]
        self.ax.clear()
        self.canvas.draw_idle()
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

        self.ax = fig.add_axes(
                [0, 0, 1, 1],
                facecolor="#222222")
        self.ax.patch.set_alpha(0.8)
        self.highlight_plot = None

    def setup_data(self):
        if not len(self.dataset.nodes):
            self.canvas.draw_idle()
            return

    def on_cluster_select(self, selected, old_selected):
        self.ax.clear()
        self.ax.patch.set_alpha(0.8)
        self.highlight_plot = None

        flattened = self.dataset.flatten(1)

        for label in selected:
            if label in self._cached_cluster_stats:
                mean, std = self._cached_cluster_stats[label]
            else:
                node = self.dataset.nodes[self.dataset.labels == label][0]
                mean = node.waveform
                std = np.std(node.waveforms, axis=0)
                self._cached_cluster_stats[label] = (mean, std)

            self.ax.fill_between(
                np.arange(len(mean)),
                mean - std,
                mean + std,
                color=self.colors[label],
                alpha=0.2,
                rasterized=True)
            self.ax.plot(
                np.arange(len(mean)),
                mean,
                color=self.colors[label],
                linewidth=3,
                rasterized=True)
        self.canvas.draw_idle()

    def on_cluster_highlight(self, new_highlight, old_highlight):
        if new_highlight is None:
            if self.highlight_plot is not None:
                self.highlight_plot.set_visible(False)
            return

        if new_highlight in self._cached_cluster_stats:
            mean, std = self._cached_cluster_stats[new_highlight]
        else:
            node = self.dataset.nodes[self.dataset.labels == new_highlight][0]
            mean = node.waveform
            std = np.std(node.waveforms, axis=0)
            self._cached_cluster_stats[new_highlight] = (mean, std)

        if new_highlight is not None and self.highlight_plot is None:
            self.highlight_plot, = self.ax.plot(
                np.arange(len(mean)),
                mean,
                color=self.colors[new_highlight],
                alpha=1.0,
                linewidth=2,
                linestyle="--"
            )
        elif new_highlight is not None and self.highlight_plot is not None:
            self.highlight_plot.set_color(self.colors[new_highlight])
            self.highlight_plot.set_ydata(mean)
            self.highlight_plot.set_visible(True)

        self.highlight_plot.set_visible(True)
        self.canvas.draw_idle()

    def init_ui(self):
        layout = widgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

