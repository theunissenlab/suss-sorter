import os
import sys
from contextlib import contextmanager
from functools import partial

import numpy as np
from PyQt5 import QtWidgets as widgets
from PyQt5.QtCore import Qt, QObject, QTimer, QThread, pyqtSignal
from PyQt5 import QtGui as gui
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

import suss.io
from suss.core import ClusterDataset

from suss.gui.cluster_select import ClusterSelector
from suss.gui.timeseries import TimeseriesPlot
from suss.gui.tsne import TSNEPlot
from suss.gui.waveforms import WaveformsPlot
from suss.gui.utils import make_color_map, get_changed_labels


class App(widgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.tsnethread = QThread()

        self.title = "SUSS Viewer"
        self.suss_viewer = None
        self.init_actions()
        self.init_ui()

    def init_actions(self):
        self.load_action = widgets.QAction("Load", self)
        self.save_action = widgets.QAction("Save", self)
        self.close_action = widgets.QAction("Exit", self)

        self.load_action.triggered.connect(self.run_file_loader)
        self.close_action.triggered.connect(self.close)

    def init_ui(self):
        self.setWindowTitle(self.title)

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("File")

        fileMenu.addAction(self.load_action)
        fileMenu.addAction(self.save_action)
        fileMenu.addAction(self.close_action)

        self.display_splash()

        rect = self.frameGeometry()
        center = widgets.QDesktopWidget().availableGeometry().center()
        rect.moveCenter(center)
        self.move(rect.topLeft())
        self.show()

    def display_splash(self):
        self.splash = Splash(self)
        self.setCentralWidget(self.splash)
        self.splash.main_button.clicked.connect(self.run_file_loader)
        self.splash.quit_button.clicked.connect(self.close)

    def display_suss_viewer(self, dataset):
        self.suss_viewer = SussViewer(dataset, self)
        self.setCentralWidget(self.suss_viewer)
        self.resize(1200, 600)
        self.show()
            
    def run_file_loader(self):
        options = widgets.QFileDialog.Options()
        options |= widgets.QFileDialog.DontUseNativeDialog
        selected_file, _ = widgets.QFileDialog.getOpenFileName(
            self,
            "Load dataset",
            ".",
            "(*.pkl)",
            options=options)
        
        if selected_file:
            self.current_file = selected_file
            self.load_dataset(selected_file)

    def run_file_saver(self):
        options = widgets.QFileDialog.Options()
        options |= widgets.QFileDialog.DontUseNativeDialog
        default_name = self.current_file.replace("sorted", "curated")
        filename, _ = widgets.QFileDialog.getSaveFileName(
            self,
            "Save dataset",
            default_name,
            "(*.pkl)",
            options=options)

        if filename:
            self.save_dataset(filename)

    def load_dataset(self, filename):
        if filename.endswith("pkl"):
            dataset = suss.io.read_pickle(filename)
        elif filename.endswith("npy"):
            dataset = suss.io.read_numpy(filename)

        if hasattr(self.suss_viewer, "tsne_plot_widget"):
            self.suss_viewer.tsne_plot_widget.reset()

        # After dataset is loaded, connect the save function
        self.save_action.triggered.connect(self.run_file_saver)
        self.display_suss_viewer(dataset)

    def save_dataset(self, filename):
        suss.io.save_pickle(filename, self.suss_viewer.dataset)
        widgets.QMessageBox.information(
                self,
                "Save",
                "Successfully saved {} to {}".format(
                    self.suss_viewer.dataset,
                    filename
                )
        )


class Splash(widgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = widgets.QVBoxLayout(self)
        self.main_button = widgets.QPushButton("Load Dataset", self)
        self.quit_button = widgets.QPushButton("Quit", self)
        self.main_button.setFixedWidth(200)
        layout.addWidget(self.main_button)
        layout.addWidget(self.quit_button)
        self.setLayout(layout)


class SussViewer(widgets.QFrame):

    # Emits the new dataset object and the old dataset object
    dataset_changed = pyqtSignal(object, object)
    selected_changed = pyqtSignal(set)

    def __init__(self, dataset=None, parent=None):
        super().__init__(parent)
        self.stack = [("load", dataset)]
        self.animation_timer = QTimer()
        self.animation_timer.start(4.0)

        self.selected = set()
        self.init_ui()
        self.setup_shortcuts()

        self.dataset_changed.connect(self.on_dataset_changed)

    @contextmanager
    def timer_paused(self):
        self.animation_timer.stop()
        yield
        self.animation_timer.start(4.0)

    def setup_shortcuts(self):
        self.undo_shortcut = widgets.QShortcut(
            gui.QKeySequence.Undo, self)
        self.undo_shortcut.activated.connect(
            self._undo)

        self.delete_shortcut = widgets.QShortcut(
            gui.QKeySequence("Backspace"), self)
        self.delete_shortcut.activated.connect(
            self.delete)

        self.delete_unselected_shortcut = widgets.QShortcut(
            gui.QKeySequence("Shift+Backspace"), self)
        self.delete_unselected_shortcut.activated.connect(
            self.delete_unselected)

        self.select_all_shortcut = widgets.QShortcut(
            gui.QKeySequence.SelectAll, self)
        self.select_all_shortcut.activated.connect(
            self.select_all)

        self.clear_shortcut = widgets.QShortcut(
            gui.QKeySequence("Ctrl+C"), self)
        self.clear_shortcut.activated.connect(
            self.clear)

        self.merge_shortcut = widgets.QShortcut(
            gui.QKeySequence("Ctrl+M"), self)
        self.merge_shortcut.activated.connect(
            self.merge)

    @property
    def dataset(self):
        return self.stack[-1][1]

    @property
    def last_action(self):
        return self.stack[-1][0]

    def _enstack(self, action, dataset):
        with self.timer_paused():
            _old_dataset = self.dataset
            self.stack.append((action, dataset))
            self.dataset_changed.emit(
                    self.dataset,
                    _old_dataset,
            )

    def _undo(self):
        if len(self.stack) == 1:
            print("Nothing left to undo.")
            return

        with self.timer_paused():
            last_action, _old_dataset = self.stack.pop()
            _new_dataset = self.dataset

            new_labels = set(self.dataset.labels)
            changed_labels = get_changed_labels(self.dataset, _old_dataset)

            if "delete unselected" in last_action:
                self.selected = set(_old_dataset.labels)
            else:
                self.selected = set.intersection(new_labels, changed_labels)
            self.colors = make_color_map(self.dataset.labels)
            self.dataset_changed.emit(
                self.dataset,
                _old_dataset
            ) 
            self.selected_changed.emit(self.selected)

    def reset(self):
        with self.timer_paused():
            _old_dataset = self.dataset
            self.stack = self.stack[:1]
            self.dataset_changed.emit(
                self.dataset,
                _old_dataset
            )
            self.dataset_updated()

    def select_all(self):
        if self.selected == set(self.dataset.labels):
            self.selected = set()
        else:
            self.selected = set(self.dataset.labels)
        self.selected_changed.emit(self.selected)

    def merge(self):
        if len(self.selected) < 2:
            widgets.QMessageBox.warning(
                    self,
                    "Merge failed",
                    "Not enough clusters selected to merge")
            return

        old_labels = set(self.dataset.labels)
        _new_dataset = self.dataset.merge_nodes(labels=self.selected)
        new_labels = set(_new_dataset.labels)
        changed_labels = get_changed_labels(_new_dataset, self.dataset)

        self.selected = set.intersection(new_labels, changed_labels)
        self.colors = make_color_map(_new_dataset.labels)
        self._enstack("merge", _new_dataset)
        self.selected_changed.emit(self.selected)
        # self.dataset_updated()

    def _delete(self, to_delete, action="delete"):
        if len(to_delete) == 0:
            widgets.QMessageBox.warning(
                    self,
                    "Delete failed",
                    "No clusters selected for deletion")
            return

        _new_dataset = self.dataset
        for label in to_delete:
            _new_dataset = _new_dataset.delete_node(label=label)

        plural = "s" if len(to_delete) > 1 else ""
        self._enstack("{} node".format(action) + plural, _new_dataset)

    def delete(self):
        self._delete(self.selected)
        self.selected = set()

    def delete_unselected(self):
        labels = set(self.dataset.labels)
        to_delete = labels - self.selected
        self.selected = set()
        self._delete(to_delete, action="delete unselected")
        self.selected_changed.emit(self.selected)

    def save(self):
        self.parent().run_file_saver()

    def load(self):
        self.parent().run_file_loader()

    def clear(self):
        self.selected = set()
        self.selected_changed.emit(self.selected)

    def toggle(self, label, selected):
        if selected and label not in self.selected:
            self.selected.add(label)
            self.selected_changed.emit(self.selected)
        elif not selected and label in self.selected:
            self.selected.remove(label)
            self.selected_changed.emit(self.selected)
        else:
            pass

    def on_dataset_changed(self):
        self.colors = make_color_map(self.dataset.labels)

    def init_ui(self):
        self.on_dataset_changed()
        # self.cluster_selector = ClusterSelector(parent=self)

        layout = widgets.QGridLayout()
        layout.setColumnStretch(1, 4)
        layout.setColumnStretch(2, 4)

        # Initialize TSNE first so it can start running TSNE in the background
        self.tsne_plot_widget = TSNEPlot(parent=self)
        layout.addWidget(self.tsne_plot_widget, 2, 2, 1, 1)
        layout.addWidget(ClusterSelector(parent=self), 1, 0, 3, 1)
        layout.addWidget(WaveformsPlot(parent=self), 2, 1, 1, 1)
        layout.addWidget(TimeseriesPlot(parent=self), 3, 1, 1, 2)

        self.setLayout(layout)


if __name__ == "__main__":
    app = widgets.QApplication(sys.argv)
    window = App()
    sys.exit(app.exec_())

