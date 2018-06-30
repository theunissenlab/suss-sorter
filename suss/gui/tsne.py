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

from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot
import time

try:
    from MulticoreTSNE import MulticoreTSNE as TSNE
except ImportError:
    from sklearn.manifold import TSNE


class BackgroundTSNE(QObject):
    finished = pyqtSignal(object)

    def __init__(self, data):
        super().__init__()
        self.data = data

    @pyqtSlot()
    def computeTSNE(self):
        print("compjutng tsne")
        tsne = TSNE(n_components=2).fit_transform(self.data)
        print("done tsne")
        self.finished.emit(tsne)


class TSNEPlot(widgets.QFrame):

    def __init__(self, size=(300, 300), parent=None):
        super().__init__(parent)
        self.size = size
        self.loading = True
        self.tsne = None

        self.setup_plots()
        self.setup_data()
        self.init_ui()
        self.parent().dataset_changed.connect(
                self.update_selected
        )
        self.parent().selected_changed.connect(
                self.update_selected
        )

    @property
    def dataset(self):
        return self.parent().dataset

    @property
    def colors(self):
        return self.parent().colors

    @property
    def selected(self):
        return self.parent().selected

    def run_tsne_background(self):
        self.loading = True
        self.worker = BackgroundTSNE(self.dataset.flatten(1).waveforms)
        self.thread = QThread()

        self.worker.finished.connect(self.update_scatter)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.computeTSNE)
        self.thread.start()

    def update_scatter(self, data):
        self.loading = False
        self.tsne = data
        self.indexes = self.dataset.flatten(1).ids
        self.update_selected(self.selected)

    def update_selected(self, selected=None):
        self.ax.clear()
        self.scatters = {}

        flattened = self.dataset.flatten(1)
        for label in self.dataset.labels:
            node = flattened.select(flattened.labels == label)
            self.scatters[label] = self.ax.scatter(
                *self.tsne[np.isin(self.indexes, node.ids)].T,
                facecolor=self.colors[label],
                edgecolor="White",

                alpha=1 if label in self.selected else 0.1,
                s=10 if label in self.selected else 5)
            # self.update_selected(self.selected)
        self.canvas.draw_idle()

    def reset(self):
        self.setup_data()

    def setup_plots(self):
        fig = Figure()
        fig.patch.set_alpha(0.0)
        self.canvas = FigureCanvas(fig)
        self.canvas.setStyleSheet("background-color:transparent;")

        self.ax = fig.add_axes(
                [0, 0, 1, 1],
                facecolor="#101010")
        self.ax.patch.set_alpha(0.8)
        # self.scat = self.ax.scatter([3, 2], [4, 1], color="White", s=20)

    def setup_data(self):
        if not len(self.dataset.nodes):
            self.canvas.draw_idle()
            return
        self.run_tsne_background()
        self.canvas.draw_idle()

    def init_ui(self):
        layout = widgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
