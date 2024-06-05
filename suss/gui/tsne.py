import functools
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
from PyQt5 import QtWidgets as widgets
from PyQt5.QtCore import Qt, QObject, QPoint, QThread, pyqtSignal, pyqtSlot
from PyQt5 import QtGui as gui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.spatial import distance

import scipy.stats
from sklearn.decomposition import PCA
import umap

from suss.gui.config import SHOW
from suss.gui.utils import require_dataset


def require_loaded(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.loading:
            return None
        return func(self, *args, **kwargs)
    return wrapper


class BackgroundTSNE(QObject):
    finished = pyqtSignal(object)

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    @pyqtSlot()
    def computeTSNE(self):
        if SHOW == "pca":
            tsne = np.hstack([
                scipy.stats.zscore(self.dataset.times[:, None]),
                PCA(n_components=2, whiten=True).fit_transform(self.dataset.waveforms)
            ])
        else:
            tsne = umap.UMAP(n_components=2).fit_transform(
                np.hstack([
                    self.dataset.times[:, None] / (60.0 * 60.0),
                    PCA(n_components=6).fit_transform(scipy.stats.zscore(self.dataset.waveforms, axis=0))
                ])
            )

        self.finished.emit(tsne[:, :2])


class TSNEPlot(widgets.QFrame):
    """Window displaying clusters on a 2D plane for easier visual selection

    Mouse hovering and clicking are routed through mpl events and
    used to update the parent (SussViewer) state.

    Computing the T-SNE embedding is fairly slow. The initial computation
    is done in the background while and displays once it is finished.
    Because it is slow, changes to clusters (mergers, deletions) do not
    trigger a re-computation of the T-SNE. This means we need to do some
    tricky indexing (see setup_data() and on_cluster_select())
    to get the labels right.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(gui.QCursor(Qt.PointingHandCursor))
        self.loading = True
        self.thread = None
        self.last_update = time.time()
        self._tsne = None
        self.main_scatter = None
        self.last_pos = None
        # For each label, the first is for selected
        # the second is for highighlighted
        self.scatters = defaultdict(list)
        self.mpl_events = []

        if len(self.dataset.flatten(2)) <= 2000:
            self.flatten_level = 2
        else:
            self.flatten_level = 1

        self.base_flattened = self.dataset.flatten(self.flatten_level)
        self.setup_plots()
        self.run_tsne_background()
        self.init_ui()

        self.parent().UPDATED_CLUSTERS.connect(self.reset)
        self.parent().CLUSTER_SELECT.connect(self.on_cluster_select)
        self.parent().CLUSTER_HIGHLIGHT.connect(self.on_cluster_highlight)
        self.window().CLOSING_DATASET.connect(self.on_close)

    @property
    def dataset(self):
        return self.parent().dataset

    @property
    def colors(self):
        return self.parent().colors

    @property
    def selected(self):
        return self.parent().selected

    @require_dataset
    def run_tsne_background(self):
        self.loading = True
        self.base_dataset = self.dataset
        self.base_flattened = self.dataset.flatten(self.flatten_level)
        self.worker = BackgroundTSNE(self.base_flattened)
        self.base_idx = self.base_flattened.ids
        self.base_labels = self.base_flattened.labels

        self._reset_thread()
        self.worker.finished.connect(self.on_tsne_completed)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.computeTSNE)
        self.thread.start()

    def on_tsne_completed(self, data):
        self._tsne = data
        self.loading = False
        self.setup_data()

    def tsne(self):
        return self._tsne[np.isin(self.base_idx, self.current_idx)]

    def on_close(self):
        if self.thread:
            self.thread.terminate()

    def _reset_thread(self):
        if self.thread:
            self.thread.terminate()
        self.thread = QThread(self)
        return self.thread

    def _set_mpl_events(self):
        self.mpl_events.append(self.canvas.mpl_connect(
            "motion_notify_event",
            self._on_hover
        ))
        self.mpl_events.append(self.canvas.mpl_connect(
            "button_press_event",
            self._on_click
        ))
        self.mpl_events.append(self.canvas.mpl_connect(
            "figure_leave_event",
            self._on_leave
        ))

    @contextmanager
    def disable_mpl_events(self):
        for mpl_event in self.mpl_events:
            self.canvas.mpl_disconnect(mpl_event)
        yield
        self._set_mpl_events()
        self.canvas.draw_idle()

    def reset(self):
        self._reset_thread()
        with self.disable_mpl_events():
            for label, scatters in self.scatters.items():
                for scat in scatters:
                    scat.remove()
                del scatters[:]
            if self.main_scatter is not None:
                self.main_scatter.remove()
                self.main_scatter = None
            self.setup_data()

    def setup_plots(self):
        fig = Figure()
        fig.patch.set_alpha(0.0)
        self.canvas = FigureCanvas(fig)
        self.canvas.setStyleSheet("background-color:transparent;")

        self.ax = fig.add_axes(
                [0.1, 0.1, 0.9, 0.9],
                facecolor="#101010")
        self.ax.patch.set_alpha(0.8)

    def setup_data(self):
        self.scatters = defaultdict(list)
        self.flattened = self.dataset.flatten(self.flatten_level)
        self.current_idx = self.flattened.ids
        self.current_labels = self.flattened.labels

        if self._tsne is None or not len(self.dataset.nodes):
            self.canvas.draw_idle()
            return

        tsne = self.tsne()

        self.main_scatter = self.ax.scatter(
            *tsne.T[:2],
            s=5,
            alpha=0.4,
            facecolors=[self.colors[label] for label in self.current_labels],
            edgecolor="White",
            rasterized=True
        )

        # for label, node in zip(self.dataset.labels, self.dataset.nodes):
        for label in self.dataset.labels:
            node = self.flattened.select(self.current_labels == label)
            # ids = node.flatten(1).ids
            ids = node.ids
            self.scatters[label].append(self.ax.scatter(
                *tsne[np.isin(self.current_idx, ids)].T[:2],
                facecolor=self.colors[label],
                edgecolor="White",
                s=14,
                alpha=1,
                rasterized=True))
            self.scatters[label][-1].set_visible(False)
            self.scatters[label].append(self.ax.scatter(
                *tsne[np.isin(self.current_idx, ids)].T[:2],
                facecolor="White",
                edgecolor=self.colors[label],
                s=20,
                alpha=1,
                rasterized=True))
            self.scatters[label][-1].set_visible(False)

        self._set_mpl_events()
        self.canvas.draw_idle()

    def init_ui(self):
        layout = widgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    @require_loaded
    def on_cluster_select(self, selected, old_selected):
        for label in self.scatters:
            scat = self.scatters[label][0]
            scat.set_visible(label in selected)
        self.ax.autoscale()
        self.canvas.draw_idle()

    @require_dataset
    @require_loaded
    def on_cluster_highlight(self, new_highlight, old_highlight, temporary):
        if old_highlight in self.scatters:
            self.scatters[old_highlight][1].set_visible(False)
        if new_highlight in self.scatters:
            self.scatters[new_highlight][1].set_visible(True)

        self.canvas.draw_idle()

    @require_loaded
    def _on_leave(self, event):
        self.parent().set_highlight(None)

    def _closest_node(self, x, y):
        """Return the index of the closest node, and the distance"""
        dist = distance.cdist([[x, y]], self.tsne())
        closest_index = dist.argmin()
        return closest_index, dist.flatten()[closest_index]

    @require_dataset
    @require_loaded
    def _on_hover(self, event):
        pos = [event.x, event.y]
        if not self.last_pos:
            self.last_pos = pos
            return

        left_vel = (
            (self.last_pos[0] - pos[0]) /
            (time.time() - self.last_update)
        )
        self.last_pos = pos
        if left_vel > 50.0 or time.time() - self.last_update < 0.1:
            # Suppress highlighting when mouse is moving left
            # (toward cluster select panel) and when the last
            # highlighting operation was within 200ms
            return

        if event.inaxes != self.ax:
            # Supress highlighting if the cursor is not in the main graph
            return
        self.last_update = time.time()
        closest_idx, dist = self._closest_node(event.xdata, event.ydata)
        if closest_idx == self.parent().highlighted:
            return
        if dist < 5.0:
            closest_label = self.current_labels[closest_idx]
            self.parent().set_highlight(closest_label)
        else:
            self.parent().set_highlight(None)

    @require_loaded
    def _on_click(self, event):
        closest_idx, dist = self._closest_node(event.xdata, event.ydata)
        label = self.current_labels[closest_idx]
        if event.button == 1:
            self.parent().toggle_selected(label, label not in self.selected)
        else:
            fig_position = self.ax.transData.transform([event.xdata, -event.ydata])
            self.parent().show_right_click_menu(
                label,
                self.mapToGlobal(QPoint(fig_position[0], fig_position[1]))
            )
