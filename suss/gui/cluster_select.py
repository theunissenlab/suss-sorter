import os
import sys
from functools import partial

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets as widgets
from PyQt5.QtCore import Qt, QObjectCleanupHandler, pyqtSignal
from PyQt5 import QtGui as gui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import ticker

from suss.analysis import align
from suss.gui.utils import clear_axes, get_changed_labels 
from suss.gui.tags import ClusterTag, UserTag

import suss.gui.config as config


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

    CARD_HEIGHT = 160 if sys.platform == "darwin" else 130

    def __init__(self, parent=None):
        super().__init__(parent)
        self.allow_scroll_to = True
        self._cached_cluster_info = {}
        self.show_auditory_responses = False

        # TOOD (kevin): make this configurable
        vocal_period_file = os.path.join(
                os.path.dirname(os.path.dirname(self.window().current_file)),
                "vocal_periods.npy")
        playback_periods_csv = os.path.join(
                os.path.dirname(os.path.dirname(self.window().current_file)),
                "playback_segments.csv")
        if os.path.exists(vocal_period_file):
            vocal_periods = np.load(vocal_period_file, allow_pickle=True)[()]
            self._stimuli = {
                "playback": [period["start_time"].item() for period in vocal_periods["playback"]],
                "live": [period["start_time"].item() for period in vocal_periods["live"]]
            }
        elif os.path.exists(playback_periods_csv):
            vocal_periods = pd.read_csv(playback_periods_csv)
            self._stimuli = {
                "playback": list(vocal_periods["start_time"]),
                "live": []
            }
        else:
            self._stimuli = None

        self.setup_data()
        self.init_ui()

        self.parent().UPDATED_CLUSTERS.connect(self.reset)
        self.parent().CLUSTER_SELECT.connect(self.update_checks)
        self.parent().CLUSTER_HIGHLIGHT.connect(self.on_cluster_highlight)
        self.parent().AUDITORY_RESPONSES.connect(self.on_auditory_responses)

    def stimuli(self, key):
        if not self.has_stimuli:
            return []
        return [period["start_time"].item() for period in self._stimuli[key]]

    @property
    def has_stimuli(self):
        return self._stimuli is not None

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

    def on_auditory_responses(self, category, state):
        if not self.show_auditory_responses and not state:
            return
        elif self.show_auditory_responses == category and state:
            return

        if not state:
            self.show_auditory_responses = False
        else:
            self.show_auditory_responses = category

        for label in list(self._cached_cluster_info.keys()):
            del self._cached_cluster_info[label]
        self.setup_data()
        self.init_ui()

    def on_cluster_highlight(self, new_highlight, old_highlight, temporary):
        if old_highlight is not None and old_highlight in self.panels:
            self.panels[old_highlight].setFrameShape(widgets.QFrame.NoFrame)

        if new_highlight is None:
            return
        new_panel = self.panels[new_highlight]
        self.panels[new_highlight].setFrameShape(widgets.QFrame.Box)
        if self.allow_scroll_to and not temporary:
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
        self.pixmaps = {}

        if not len(self.dataset):
            return

        wf_ylims = (np.min(self.dataset.waveforms), np.max(self.dataset.waveforms))

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

        for _progress, label_idx in enumerate(ordered_idx):
            cluster_label = self.dataset.labels[label_idx]
            cluster = self.dataset.nodes[label_idx]

            header = widgets.QHBoxLayout()
            header_label = widgets.QLabel(
                "<b>{}</b> (n={}) {} clusters".format(
                    cluster_label,
                    cluster.count,
                    "No cluster data" if not "nodes" in cluster._data else len(cluster.nodes)
                )
            )
            header_label.setFixedHeight(14)
            check_button = create_check_button(" ")
            check_button.clicked[bool].connect(partial(self.toggle, cluster_label))
            check_button.hover.connect(partial(self.set_highlight, cluster_label))
            self.buttons[cluster_label] = check_button
            header.addWidget(check_button)
            header.addWidget(header_label)
            header_label.setStyleSheet("""
                QLabel {
                    border:none;
                    padding:0;
                    margin:0;
                    font-size: 12px;
                }
            """)

            header.setAlignment(Qt.AlignTop)

            pixmap = gui.QPixmap(10, self.CARD_HEIGHT)
            pixmap.fill(gui.QColor(*[
                255 * c
                for c in self.colors[cluster_label]
            ]))
            self.pixmaps[cluster_label] = pixmap
            color_banner = widgets.QLabel()
            color_banner.setPixmap(pixmap)
            # color_banner.setScaledContents(True)

            container = widgets.QFrame(self) # ClusterClickFilter(self, label=cluster_label)
            # container.setStyleSheet("QWidget#highlighted {border: 2px dotted black;}")
            cluster_layout = widgets.QGridLayout()
            # cluster_layout.setColumnStretch(0, 1)
            cluster_layout.setColumnStretch(1, 6)

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
                plots_widget.set_ylim(wf_ylims)
            if not plots_loaded:
                plots_widget = ClusterInfo(
                        cluster,
                        self.colors[cluster_label],
                        ylim=wf_ylims,
                        parent=self)
                self._cached_cluster_info[cluster_label] = plots_widget
                cluster_layout.addWidget(plots_widget, 1, 1)

            container.setLayout(cluster_layout)

            container.setContextMenuPolicy(Qt.CustomContextMenu)
            container.customContextMenuRequested.connect(
                partial(self.on_click, container, cluster, cluster_label)
            )
            container.setFixedHeight(self.CARD_HEIGHT)

            self.layout.addWidget(container)
            self.panels[cluster_label] = container
            progress.setValue(_progress)

        progress.setValue(len(self.dataset.nodes) + 1)
        progress.hide()

    def create_tags_menu(self, cluster):
        menu = widgets.QMenu("Tags")

        for _tag in ClusterTag:
            _checkbox = widgets.QCheckBox(_tag.name, menu)
            _act = widgets.QWidgetAction(menu)
            _act.setDefaultWidget(_checkbox)
            # _act.setCheckable(True)
            _checkbox.setChecked(_tag in cluster.tags)
            _checkbox.toggled.connect(partial(self._update_tag, cluster, _tag))
            menu.addAction(_act)

        for _tag in UserTag:
            _checkbox = widgets.QCheckBox(_tag.name, menu)
            _act = widgets.QWidgetAction(menu)
            _act.setDefaultWidget(_checkbox)
            # _act.setCheckable(True)
            _checkbox.setChecked(_tag in cluster.tags)
            _checkbox.toggled.connect(partial(self._update_tag, cluster, _tag))
            menu.addAction(_act)

        return menu

    def _update_tag(self, cluster, _tag, state):
        if state and _tag not in cluster.tags:
            cluster.add_tag(_tag)
        elif not state and _tag in cluster.tags:
            cluster.remove_tag(_tag)

    def on_click(self, obj, cluster, label, pos):
        self.parent().show_right_click_menu(
                label,
                obj.mapToGlobal(pos),
                [self.create_tags_menu(cluster)]
        )

    def init_ui(self):
        self.frame = widgets.QGroupBox()
        self.frame.setLayout(self.layout)
        self.setWidget(self.frame)
        self.setWidgetResizable(True)
        self.setFixedWidth(360)


class ClusterInfo(widgets.QWidget):

    def __init__(self, cluster, color, size=(265, 75), ylim=None, parent=None):
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
        self.ax_wf = fig.add_axes([0.0, 0, 0.33, 1])
        self.ax_wf.patch.set_alpha(0.0)
        self.ax_isi = fig.add_axes([0.33, 0, 0.33, 1])
        self.ax_isi.patch.set_alpha(0.0)
        self.ax_psth = fig.add_axes([0.66, 0, 0.34, 1])
        self.ax_psth.patch.set_alpha(0.0)
        self.fr_label = self.ax_wf.text(0, self.ax_wf.get_ylim()[0], "",
                horizontalalignment="left", verticalalignment="bottom", fontsize=6)
        self.snr_label = None
        clear_axes(self.ax_isi, self.ax_psth)

    def set_ylim(self, ylim):
        self.ylim = ylim
        self.ax_wf.set_ylim(*self.ylim)
        self.fr_label.set_y(self.ylim[0])
        if self.snr_label:
            self.snr_label.set_y(self.ylim[0])

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

        fr = len(cluster) / (np.max(cluster.times) - np.min(cluster.times))

        snr = (np.max(mean) - np.min(mean)) / np.mean(std)
        # snr = np.abs(mean)[len(mean) // 2] / std[len(mean) // 2]
        if not self.snr_label:
            self.snr_label = self.ax_wf.text(self.ax_wf.get_xlim()[1], self.ax_wf.get_ylim()[0], "",
                    horizontalalignment="right", verticalalignment="bottom", fontsize=6)
        self.snr_label.set_text("SNR: {:.1f}".format(snr))
        self.ax_wf.yaxis.set_major_locator(ticker.MultipleLocator(base=100))
        self.ax_wf.xaxis.set_major_locator(ticker.MultipleLocator(base=10))
        self.ax_wf.grid(True)

        if self.ylim is None:
            self.set_ylim(self.ax_wf.get_ylim())
        else:
            self.set_ylim(self.ylim)

        self.fr_label.set_text("{:.1f} uV\n{:.1f} Hz".format(mean[len(mean) // 2], fr))

        isi = np.diff(cluster.times)
        isi_violations = np.sum(isi < 0.001) / len(isi)
        n_bins = config.ISI_BINS
        t_max = config.ISI_MAX
        hist, bin_edges = np.histogram(isi, bins=n_bins, density=True, range=(0, t_max))
        self.ax_isi.bar(
                (bin_edges[:-1] + bin_edges[1:]) / 2,
                hist,
                width=t_max / n_bins,
                color="Black")
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

        if self.parent().show_auditory_responses and self.parent().has_stimuli:
            stimuli_times = [
                start_time for start_time in self.parent().stimuli(
                    self.parent().show_auditory_responses
                )
                if cluster.times[0] <= start_time < cluster.times[-1]
            ]
            psth, _ = align(cluster.flatten(), stimuli_times, -1, 1)
            if len(psth):
                hist, bin_edges = np.histogram(
                        np.concatenate(psth),
                        bins=20,
                        density=False,
                        range=(0, 1))
                self.ax_psth.bar(
                        (bin_edges[:-1] + bin_edges[1:]) / 2,
                        hist / len(psth),
                        width=1 / 20.0,
                        color="Black")
                self.ax_psth.vlines(
                        [0.5],
                        *self.ax_psth.get_ylim(),
                        color="Red",
                        linestyle="--",
                        linewidth=0.5,
                )

        self.canvas.draw_idle()

    def init_ui(self):
        layout = widgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
