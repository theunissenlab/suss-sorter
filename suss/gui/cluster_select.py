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
from matplotlib import ticker

from suss.gui.utils import make_color_map, clear_axes, get_changed_labels


class HoverButton(widgets.QPushButton):
    hover = pyqtSignal(bool)

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setMouseTracking(True)

    def enterEvent(self, QEvent):
        self.hover.emit(True)

    def leaveEvent(self, QEvent):
        self.hover.emit(False)


def create_check_button(text):
    button = HoverButton(text)
    button.setCheckable(True)
    button.setDefault(False)
    button.setAutoDefault(False)

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
        self.allow_scroll_to = True
        self._cached_cluster_info = {}
        self.setup_data()
        self.init_ui()

        self.parent().UPDATED_CLUSTERS.connect(self.reset)
        self.parent().CLUSTER_SELECT.connect(self.update_checks)
        self.parent().CLUSTER_HIGHLIGHT.connect(self.on_cluster_highlight)

    def reset(self, new_dataset, old_dataset):
        old_scroll = self.verticalScrollBar().value()
        for label in get_changed_labels(new_dataset, old_dataset):
            if label in self._cached_cluster_info:
                del self._cached_cluster_info[label]
        self.setup_data()
        self.init_ui()
        for label, info in self._cached_cluster_info.items():
            info.update_color(self.colors[label])
        for label in self.dataset.labels:
            self.pixmaps[label].fill(gui.QColor(*[
                255 * c
                for c in self.colors[label]
            ]))

        self.verticalScrollBar().setValue(old_scroll)

    @property
    def dataset(self):
        return self.parent().dataset

    @property
    def colors(self):
        return self.parent().colors

    def toggle(self, label, selected):
        self.parent().toggle_selected(label, selected)

    def update_checks(self, selected):
        for label, button in self.buttons.items():
            button.clicked.emit(label in selected)
            button.setChecked(label in selected)

    def on_cluster_highlight(self, new_highlight, old_highlight):
        if old_highlight and old_highlight in self.panels:
            self.panels[old_highlight].setFrameShape(widgets.QFrame.NoFrame)

        if new_highlight is None:
            return
        new_panel = self.panels[new_highlight]
        self.panels[new_highlight].setFrameShape(widgets.QFrame.Box)
        if self.allow_scroll_to:
            self.verticalScrollBar().setValue(new_panel.y())

    def set_highlight(self, cluster_label, selected):
        # Hacky flag to prevent jumping the scroll bar
        # when highlighted through the cluster select menu
        self.allow_scroll_to = not selected
        self.parent().set_highlight(cluster_label if selected else None)

    def setup_data(self):
        self.buttons = {}
        self.panels = {}

        ordered_idx = reversed(np.argsort(self.dataset.labels))

        self.layout = widgets.QVBoxLayout(self)

        progress = widgets.QProgressDialog(
                "Loading {} clusters".format(
                    len(self.dataset.nodes)
                ),
                None,
                0,
                len(self.dataset.nodes) + 1,
                self)
        progress.setMinimumDuration(2000)
        progress.open()
        self.pixmaps = {}

        wf_ylims = (np.min(self.dataset.waveforms), np.max(self.dataset.waveforms))

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
            check_button = create_check_button(" ")
            check_button.clicked[bool].connect(partial(self.toggle, cluster_label))
            check_button.hover.connect(partial(self.set_highlight, cluster_label))
            self.buttons[cluster_label] = check_button
            header.addWidget(check_button)
            header.addWidget(header_label)

            pixmap = gui.QPixmap(8, 60)
            pixmap.fill(gui.QColor(*[
                255 * c
                for c in self.colors[cluster_label]
            ]))
            self.pixmaps[cluster_label] = pixmap
            color_banner = widgets.QLabel()
            color_banner.setPixmap(pixmap)
            color_banner.setScaledContents(True)

            container = widgets.QFrame(self)
            # container.setStyleSheet("QWidget#highlighted {border: 2px dotted black;}")
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
                        ylim=wf_ylims,
                        parent=self)
                self._cached_cluster_info[cluster_label] = plots_widget
                cluster_layout.addWidget(plots_widget, 1, 1)

            container.setLayout(cluster_layout)

            self.layout.addWidget(container)
            self.panels[cluster_label] = container
            progress.setValue(_progress)

        progress.setValue(len(self.dataset.nodes) + 1)

    def init_ui(self):
        self.frame = widgets.QGroupBox()
        self.frame.setLayout(self.layout)
        self.setWidget(self.frame)
        self.setWidgetResizable(True)
        self.setFixedWidth(350)


class ClusterInfo(widgets.QWidget):

    def __init__(self, cluster, color, size=(200, 75), ylim=None, parent=None):
        super().__init__(parent)
        self.cluster = cluster
        self.color = color
        self.size = size
        self.ylim = ylim

        self.setup_plots()
        self.setup_data()
        self.init_ui()

    def update_color(self, color):
        self.color = color
        self.wf_plot.set_color(self.color)
        self.canvas.draw_idle()

    def setup_plots(self):
        fig = Figure()
        fig.patch.set_alpha(0.0)
        self.canvas = FigureCanvas(fig)
        self.canvas.setFixedSize(*self.size)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.ax_wf = fig.add_axes([0.0, 0, 0.5, 1])
        self.ax_wf.patch.set_alpha(0.0)
        self.ax_isi = fig.add_axes([0.5, 0, 0.5, 1])
        self.ax_isi.patch.set_alpha(0.0)
        clear_axes(self.ax_isi)

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
        self.wf_plot = self.ax_wf.plot(
                mean,
                color=self.color,
                alpha=1.0,
                linewidth=2)[0]

        self.ax_wf.yaxis.set_major_locator(ticker.MultipleLocator(base=100))
        self.ax_wf.xaxis.set_major_locator(ticker.MultipleLocator(base=10))
        self.ax_wf.grid(True)

        if self.ylim is None:
            self.ylim = self.ax_wf.get_ylim()
        else:
            self.ax_wf.set_ylim(*self.ylim)

        isi = np.diff(cluster.times)
        isi_violations = np.sum(isi < 0.001) / len(isi)
        n_bins = 40
        t_max = 0.2
        hist, bin_edges = np.histogram(isi, bins=n_bins, density=True, range=(0, t_max))
        self.ax_isi.bar((bin_edges[:-1] + bin_edges[1:]) / 2, hist, width=t_max / n_bins, color="Black")
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
                linewidth=0.2)
        self.ax_isi.set_xlim(0, t_max)

        peaks = np.min(cluster.waveforms, axis=1)
        hist, bin_edges = np.histogram(peaks, bins=50, density=True, range=(-200, 0))

        self.canvas.draw_idle()

    def init_ui(self):
        layout = widgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

