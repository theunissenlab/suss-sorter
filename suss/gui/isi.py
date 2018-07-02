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


class ISIPlot(widgets.QFrame):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_plots()
        self.setup_data()
        self.init_ui()
        self.parent().UPDATED_CLUSTERS.connect(self.reset)
        self.parent().CLUSTER_HIGHLIGHT.connect(self.on_cluster_highlight)
        self.parent().CLUSTER_SELECT.connect(self.on_cluster_select)

    def reset(self, new_dataset, old_dataset):
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
        fig = Figure(facecolor="#C0C0C0")
        fig.patch.set_alpha(1.0)
        self.canvas = FigureCanvas(fig)
        self.canvas.setStyleSheet("background-color:transparent;")

        self.ax = fig.add_axes(
                [0, 0.15, 1, 0.85],
                facecolor="#111111")
        self.ax.patch.set_alpha(0.8)
        self.ax.set_xlim(0, 0.03)
        self.ax.set_xticks([0.001, 0.02])
        self.ax.set_xticklabels(["1ms", "20ms"],
                horizontalalignment="center",
                fontsize=5)
        for tick in self.ax.get_xaxis().get_major_ticks():
            tick.set_pad(0)

        self.text_ax = fig.add_axes(
                [0, 0, 1, 1],
                xlim=(0, 1),
                ylim=(0, 1))
        self.text_ax.patch.set_alpha(0.0)
        self.isi_label = self.text_ax.text(
            0.98,
            0.95,
            "",
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=8,
            color="White")

    def setup_data(self):
        if not len(self.dataset.nodes):
            self.canvas.draw_idle()
            return

    def on_cluster_select(self, selected, old_selected):
        self.ax.clear()
        self.ax.set_xlim(0, 0.03)
        self.ax.set_xticks([0.001, 0.02])
        self.ax.set_xticklabels(["1ms", "20ms"],
                horizontalalignment="left",
                fontsize=5)
        for tick in self.ax.get_xaxis().get_major_ticks():
            tick.set_pad(0)

        self.ax.patch.set_alpha(0.8)

        clusters = self.dataset.select(
            np.isin(self.dataset.labels, list(selected))
        ).flatten()

        isi = np.diff(clusters.times)
        if not len(isi):
            self.canvas.draw_idle()
            return

        isi_violations = len(np.where(isi < 0.001)) / len(isi)

        across_clusters = clusters.labels[:-1] != clusters.labels[1:]
        within_cluster = clusters.labels[:-1] == clusters.labels[1:]

        if not np.sum(across_clusters) + np.sum(within_cluster):
            self.canvas.draw_idle()
            return

        self.ax.hist(
            [
                isi[across_clusters],
                isi[within_cluster]
            ],
            bins=30,
            density=True,
            range=(0, 0.03),
            stacked=True,
            alpha=0.8,
            color=["Orange", "Black"]
        )
        self.ax.vlines(
                0.001,
                *self.ax.get_ylim(),
                color="Red",
                linestyle="--",
                linewidth=0.5)

        self.isi_label.set_text(
            "{:.1f}%\nISI violations".format(
                100.0 * isi_violations
            )
        )

        self.canvas.draw_idle()

    def on_cluster_highlight(self, new_highlight, old_highlight):
        pass

    def init_ui(self):
        layout = widgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

