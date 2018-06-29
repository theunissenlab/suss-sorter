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

from suss.gui.utils import make_color_map, clear_axes


class TimeseriesPlot(widgets.QFrame):

    def __init__(self, size=(700, 100), parent=None):
        super().__init__(parent)
        self.size = size

        self.setup_plots()
        self.setup_data()
        self.init_ui()
        self.parent().dataset_changed.connect(
                self.reset
        )
        self.parent().selected_changed.connect(
                self.update_selected
        )

    def reset(self):
        # Delete the old layout
        # widgets.QWidget().setLayout(self.layout())
        for _, scatters in self.scatters.items():
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

        self.ax1 = fig.add_axes(
                [0, 0.1, 1, 0.44],
                facecolor="#111111")
        self.ax1.set_yticks([])
        self.ax1.patch.set_alpha(0.8)

        self.ax2 = fig.add_axes(
                [0, 0.55, 1, 0.44],
                facecolor="#111111")
        self.ax2.set_yticks([])
        self.ax2.set_xticks([])
        self.ax2.patch.set_alpha(0.8)

        self.axes = [self.ax1, self.ax2]

        self.scatters = defaultdict(list)
        self.main_scatters = []

        self._frame = 0
        self.timer = QTimer()
        # TODO (kevin): make the speed variable
        self.timer.start(4.0)
        self.timer.timeout.connect(self.rotate)

    def rotate(self):
        # TODO (kevin): make the speed variable
        if not len(self.dataset.nodes):
            return

        _t = (2 * np.pi) * (self._frame % 200) / 200
        for dim in range(2):
            self.main_scatters[dim].set_offsets(
                np.array([
                    self.dataset.flatten(1).times,
                    np.cos(_t + dim * np.pi / 2) * self.pcs.T[0] + np.sin(_t + dim * np.pi / 2) * self.pcs.T[dim + 1]
                ]).T
            )
        for label, node in zip(self.dataset.labels, self.dataset.nodes):
            pcs = self.pcs[self.flattened.labels == label]
            for dim in range(2):
                self.scatters[label][dim].set_offsets(
                    np.array([
                        node.times,
                        np.cos(_t + dim * np.pi / 2) * pcs.T[0] + np.sin(_t + dim * np.pi / 2) * pcs.T[dim + 1]
                    ]).T
                )
        self._frame += 1
        self.canvas.draw_idle()

    def setup_data(self):
        if not len(self.dataset.nodes):
            self.canvas.draw_idle()
            return

        self.flattened = self.dataset.flatten(1)

        self.pca = PCA(n_components=3).fit(self.dataset.flatten().waveforms)
        self.pcs = self.pca.transform(self.flattened.waveforms)

        self.main_scatters.append(self.ax1.scatter(
            self.flattened.times,
            self.pcs.T[0],
            s=5,
            alpha=0.8,
            color="Gray",
            rasterized=True
        ))
        self.main_scatters.append(self.ax2.scatter(
            self.flattened.times,
            self.pcs.T[1],
            s=5,
            alpha=0.8,
            color="Gray",
            rasterized=True
        ))
        ylim = max(*self.ax1.get_ylim())
        self.ax1.set_ylim(-ylim, ylim)
        self.ax2.set_ylim(-ylim, ylim)

        for label, node in zip(self.dataset.labels, self.dataset.nodes):
            pcs = self.pcs[self.flattened.labels == label]
            for dim in range(2):
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
        self.canvas.draw_idle()

    def update_selected(self, selected=None):
        if selected is None:
            selected = set()
        for label in self.scatters:
            for scat in self.scatters[label]:
                scat.set_visible(label in selected)
        self.canvas.draw_idle()

    def init_ui(self):
        layout = widgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

