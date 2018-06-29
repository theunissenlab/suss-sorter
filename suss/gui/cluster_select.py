import os
import sys
from functools import partial
import time

import numpy as np
from PyQt5 import QtWidgets as widgets
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5 import QtGui as gui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from suss.gui.utils import make_color_map, clear_axes, get_changed_labels


def create_check_button(text, color, cb):
    button = widgets.QPushButton(text)
    button.setCheckable(True)
    button.setDefault(False)
    button.setAutoDefault(False)
    button.clicked[bool].connect(cb)

    def flip_text(selected):
        if selected:
            button.setText("X")
        else:
            button.setText(" ")

    button.clicked[bool].connect(flip_text)
    color = "white"
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
    button.setFixedSize(20, 20)

    return button


class ClusterSelector(widgets.QScrollArea):

    def __init__(self, parent=None):
        super().__init__(parent)
        # FIXME (kevin): need to cache bust colors!
        self._cached_cluster_info = {}
        self.setup_data()
        self.init_ui()

        self.parent().dataset_changed.connect(
            self.reset
        )
        self.parent().selected_changed.connect(
            self.update_checks
        )

    def reset(self, new_dataset, old_dataset):
        old_scroll = self.verticalScrollBar().value()
        for label in get_changed_labels(new_dataset, old_dataset):
            if label in self._cached_cluster_info:
                del self._cached_cluster_info[label]
        self.setup_data()
        self.init_ui()
        self.verticalScrollBar().setValue(old_scroll)

    @property
    def dataset(self):
        return self.parent().dataset

    @property
    def colors(self):
        return self.parent().colors

    def toggle(self, label, selected):
        self.parent().toggle(label, selected)

    def update_checks(self, selected):
        for label, button in self.buttons.items():
            button.clicked.emit(label in selected)
            button.setChecked(label in selected)

    def setup_data(self):
        self.buttons = {}

        ordered_idx = reversed(np.argsort(self.dataset.labels))

        self.layout = widgets.QVBoxLayout(self)

        progress = widgets.QProgressDialog(
                "Loading {} clusters".format(
                    len(self.dataset.nodes)
                ),
                None,
                0,
                len(self.dataset.nodes),
                self)
        progress.setMinimumDuration(2000)
        progress.open()

        for _progress, label_idx in enumerate(ordered_idx):
            cluster_label = self.dataset.labels[label_idx]
            cluster = self.dataset.nodes[label_idx]

            header = widgets.QHBoxLayout()
            header_label = widgets.QLabel(
                "<b>{}</b> (n={}) ({} subclusters)".format(
                    cluster_label,
                    cluster.waveform_count,
                    len(cluster.nodes)
                )
            )
            check_button = create_check_button(
                " ",
                self.colors[cluster_label],
                partial(self.toggle, cluster_label))
            self.buttons[cluster_label] = check_button
            header.addWidget(check_button)
            header.addWidget(header_label)

            pixmap = gui.QPixmap(10, 60)
            pixmap.fill(gui.QColor(*[
                255 * c
                for c in self.colors[cluster_label]
            ]))
            color_banner = widgets.QLabel()
            color_banner.setPixmap(pixmap)
            color_banner.setScaledContents(True)

            container = widgets.QWidget(self)
            cluster_layout = widgets.QGridLayout()
            cluster_layout.addWidget(color_banner, 0, 0, 2, 1)
            cluster_layout.addLayout(header, 0, 1)

            # Try to access a cached version of the cluster info plots
            plots_loaded = False
            if cluster_label in self._cached_cluster_info:
                plots_widget = self._cached_cluster_info[cluster_label]
                try:
                    cluster_layout.addWidget(plots_widget, 1, 1)
                except:
                    del self._cached_cluster_info[cluster_label]
                else:
                    plots_loaded = True
            if not plots_loaded:
                plots_widget = ClusterInfo(
                        cluster,
                        self.colors[cluster_label],
                        parent=self)
                self._cached_cluster_info[cluster_label] = plots_widget
                cluster_layout.addWidget(plots_widget, 1, 1)

            container.setLayout(cluster_layout)

            self.layout.addWidget(container)
            progress.setValue(_progress)

        progress.setValue(len(self.dataset.nodes))

    def init_ui(self):
        self.frame = widgets.QGroupBox()
        self.frame.setLayout(self.layout)
        self.setWidget(self.frame)
        self.setWidgetResizable(True)
        self.setFixedWidth(420)


class ClusterInfo(widgets.QWidget):

    def __init__(self, cluster, color, size=(300, 75), parent=None):
        super().__init__(parent)
        self.cluster = cluster
        self.color = color
        self.size = size

        self.setup_plots()
        self.setup_data()
        self.init_ui()

    def setup_plots(self):
        fig = Figure()
        fig.patch.set_alpha(0.0)
        self.canvas = FigureCanvas(fig)
        self.canvas.setFixedSize(*self.size)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.ax_wf = fig.add_axes([0.0, 0, 0.32, 1])
        self.ax_wf.patch.set_alpha(0.0)
        self.ax_isi = fig.add_axes([0.33, 0, 0.32, 1])
        self.ax_isi.patch.set_alpha(0.0)
        self.ax_skew = fig.add_axes([0.66, 0, 0.32, 1])
        self.ax_skew.patch.set_alpha(0.0)
        clear_axes(self.ax_wf, self.ax_isi, self.ax_skew)

    def setup_data(self):
        cluster = self.cluster.flatten()

        mean = np.mean(self.cluster.waveforms, axis=0)
        std = np.std(cluster.waveforms, axis=0)
        self.ax_wf.fill_between(
                np.arange(len(mean)),
                mean - std,
                mean + std,
                color="Black",
                alpha=0.25)
        self.ax_wf.plot(
                mean,
                color=self.color,
                alpha=1.0,
                linewidth=2)
        self.ax_wf.set_ylim(-150, 100)
        self.ax_wf.hlines(
                [-100, -50, 0, 50],
                *self.ax_wf.get_xlim(),
                color="Black",
                linestyle="--",
                linewidth=0.5,
                alpha=0.5)
        self.ax_wf.vlines(
                [10, 20, 30],
                *self.ax_wf.get_ylim(),
                color="Black",
                linestyle="--",
                linewidth=0.5,
                alpha=0.5)

        isi = np.diff(cluster.times)
        isi_violations = np.sum(isi < 0.001) / len(isi)
        hist, bin_edges = np.histogram(isi, bins=30, density=True, range=(0, 0.03))
        self.ax_isi.bar(bin_edges[:-1] + 0.0005, hist, width=0.001, color="Black")
        '''
        self.ax_isi.hist(
                isi,
                bins=50,
                range=(0, 0.05),
                density=True,
                color="Black")
        '''
        self.ax_isi.text(
                self.ax_isi.get_xlim()[1] * 0.9,
                self.ax_isi.get_ylim()[1] * 0.9,
                "{:.1f}% < 1ms".format(100.0 * isi_violations),
                fontsize=6,
                horizontalalignment="right",
                verticalalignment="top")
        self.ax_isi.vlines(
                0.001,
                *self.ax_isi.get_ylim(),
                color="Red",
                linestyle="--",
                linewidth=0.5)
        self.ax_isi.set_xlim(0, 0.03)

        peaks = np.min(cluster.waveforms, axis=1)
        hist, bin_edges = np.histogram(peaks, bins=20, density=True, range=(-200, 0))
        self.ax_skew.bar(bin_edges[:-1] + 5, hist, width=10, color="Black")
        '''
        self.ax_skew.hist(peaks,
                bins=20,
                range=(-200, 0),
                density=True,
                color="Black")
        '''
        self.ax_skew.vlines(
                [-100, -50],
                *self.ax_skew.get_ylim(),
                color="Black",
                linestyle=":",
                alpha=0.2)
        self.canvas.draw_idle()
        # self.canvas.mpl_connect("button_press_event", self._on_press)

    # def _on_press(self, event):
    #     print(event)

    def init_ui(self):
        layout = widgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

