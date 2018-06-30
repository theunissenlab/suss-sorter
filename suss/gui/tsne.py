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
from scipy.spatial import distance

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
        tsne = TSNE(n_components=2).fit_transform(self.data)
        print("Computed TSNE")
        self.finished.emit(tsne)


class TSNEPlot(widgets.QFrame):


    def __init__(self, size=(300, 300), parent=None):
        super().__init__(parent)
        self.setCursor(gui.QCursor(Qt.PointingHandCursor))

        self.size = size
        self.loading = True
        self._tsne = None
        self.last_highlight = None
        self.last_highlight_node = None
        self.mpl_events = []

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
        self.flattened = self.dataset.flatten(1)
        self.worker = BackgroundTSNE(self.flattened.waveforms)
        self.original_index = self.flattened.ids

        self.worker.finished.connect(self.update_scatter)
        self.worker.moveToThread(self.window().tsnethread)
        self.window().tsnethread.started.connect(self.worker.computeTSNE)
        self.window().tsnethread.start()

    def update_scatter(self, data):
        self._tsne = data
        self.loading = False
        self.flattened = self.dataset.flatten(1)
        self.indexes = self.flattened.ids
        self.labels = self.flattened.labels
        self.update_selected(self.selected)

    @property
    def tsne(self):
        return self._tsne[np.isin(self.original_index, self.indexes)]

    def update_selected(self, selected=None):
        self.ax.clear()
        self.scatters = {}
        for mpl_event in self.mpl_events:
            self.canvas.mpl_disconnect(mpl_event)
        self.mpl_events = []

        self.flattened = self.dataset.flatten(1)
        self.indexes = self.flattened.ids
        self.labels = self.flattened.labels
        for label in self.dataset.labels:
            node = self.flattened.select(self.flattened.labels == label)
            self.scatters[label] = self.ax.scatter(
                *self.tsne[np.isin(self.indexes, node.ids)].T,
                facecolor=self.colors[label],
                edgecolor="White",
                alpha=1 if label in self.selected else 0.2,
                s=14 if label in self.selected else 5)
            # self.update_selected(self.selected)
        self.canvas.draw_idle()

        self.mpl_events.append(
            self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        )
        self.mpl_events.append(
            self.canvas.mpl_connect("button_press_event", self._on_click)
        )
        self.mpl_events.append(
            self.canvas.mpl_connect("figure_leave_event", self._on_leave)
        )

    def _on_leave(self, event):
        self._clear_last_highlight()

    def _closest_node(self, x, y):
        closest_index = distance.cdist([[x, y]], self.tsne).argmin()
        return closest_index

    def _clear_last_highlight(self):
        if self.last_highlight and self.last_highlight in self.scatters:
            self.scatters[self.last_highlight].set_sizes([
                (14 if self.last_highlight in self.selected else 5)
                for _ in self.last_highlight_node.waveforms
            ])
            self.scatters[self.last_highlight].set_alpha(
                1 if self.last_highlight in self.selected else 0.2
            )
        self.last_highlight = None
        self.last_highlight_node = None
        self.canvas.draw_idle()

    def _on_hover(self, event):
        closest_idx = self._closest_node(event.xdata, event.ydata)
        closest_label = self.labels[closest_idx]

        self._clear_last_highlight()

        node = self.flattened.select(self.flattened.labels == closest_label)
        self.last_highlight = closest_label
        self.last_highlight_node = node
        self.scatters[closest_label].set_sizes([25 for _ in node.waveforms])
        self.scatters[closest_label].set_alpha(1)
        self.canvas.draw_idle()

    def _on_click(self, event):
        if not self.last_highlight:
            return
        self.parent().toggle(
            self.last_highlight,
            self.last_highlight not in self.selected
        )
        self._on_hover(event)

    def reset(self):
        self.ax.clear()
        for mpl_event in self.mpl_events:
            self.canvas.mpl_disconnect(mpl_event)
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
