import os
import sys
from contextlib import contextmanager
from functools import partial

from PyQt5 import QtWidgets as widgets
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5 import QtGui as gui

import suss.io

import suss.gui.config as config
from suss.gui.cluster_select import ClusterSelector
from suss.gui.isi import ISIPlot
from suss.gui.projections import ProjectionsPlot
from suss.gui.timeseries import TimeseriesPlot
from suss.gui.tsne import TSNEPlot
from suss.gui.waveforms import WaveformsPlot
from suss.gui.utils import make_color_map, get_changed_labels
from suss.operations import (
        add_nodes,
        delete_nodes,
        match_one,
        merge_nodes,
        recluster_node
)


class App(widgets.QMainWindow):

    CLOSING_DATASET = pyqtSignal()
    LOADED_DATASET = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._closing = False
        self.title = "SUSS Viewer"
        self.suss_viewer = None
        self.init_actions()
        self.init_ui()
        self.setup_shortcuts()

    def setup_shortcuts(self):
        self.save_action.setShortcut(gui.QKeySequence.Save)
        self.addAction(self.save_action)

        self.load_action.setShortcut(gui.QKeySequence.Open)
        self.addAction(self.load_action)

        self.close_action.setShortcut(gui.QKeySequence.Close)
        self.addAction(self.close_action)

        self.addAction(self.load_action)
        self.addAction(self.save_action)
        self.addAction(self.close_action)

    def closeEvent(self, event):
        skip_confirmation = (
                # Workaround for PyQt closeEvent being called twice bug
                # See: https://bugreports.qt.io/browse/QTBUG-43344
                self._closing == True or
                not self.suss_viewer or
                len(self.suss_viewer.stack) == 1
        )
        if skip_confirmation:
            event.accept()
            return

        quit_msg = "Are you sure you want to quit?\nMake sure any progress is saved."
        reply = widgets.QMessageBox.question(
                self,
                'Message',
                quit_msg,
                widgets.QMessageBox.Yes,
                widgets.QMessageBox.No)

        if reply == widgets.QMessageBox.Yes:
            self._closing = True
            event.accept()
        else:
            event.ignore()

    def init_actions(self):
        self.load_action = widgets.QAction("Open...", self)
        self.save_action = widgets.QAction("Save", self)
        self.close_action = widgets.QAction("Close", self)

        self.load_action.triggered.connect(self.run_file_loader)
        self.save_action.triggered.connect(self.run_file_saver)
        self.close_action.triggered.connect(self.close)

    def init_ui(self):
        self.setWindowTitle(self.title)

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("&File")

        fileMenu.addAction(self.load_action)
        fileMenu.addSeparator()
        fileMenu.addAction(self.save_action)
        # fileMenu.addAction(self.close_action)

        self.display_splash()

    def _center_on(self, w):
        rect = w.frameGeometry()
        geom = widgets.QDesktopWidget().availableGeometry()
        center_on = geom.center()
        center_on.setY(center_on.y() - geom.height() / 8)
        rect.moveCenter(center_on)
        self.move(rect.topLeft())

    def display_splash(self):
        self.splash = Splash(self)
        self.setCentralWidget(self.splash)
        self.splash.main_button.clicked.connect(self.run_file_loader)
        self.splash.quit_button.clicked.connect(self.close)
        self._center_on(self.splash)
        self.show()

    def display_suss_viewer(self, dataset):
        if self.suss_viewer:
            self.CLOSING_DATASET.emit()
        self.suss_viewer = SussViewer(dataset, self)
        self.setCentralWidget(self.suss_viewer)
        self._center_on(self.suss_viewer)
        self.show()

    def run_file_loader(self):
        options = widgets.QFileDialog.Options()
        # options |= widgets.QFileDialog.DontUseNativeDialog
        selected_file, _ = widgets.QFileDialog.getOpenFileName(
            self,
            "Load dataset",
            config.BASE_DIRECTORY or ".",
            "(*.pkl)",
            options=options)

        if selected_file:
            self.current_file = selected_file
            self.load_dataset(selected_file)

    def run_file_saver(self):
        if not self.suss_viewer:
            return

        options = widgets.QFileDialog.Options()
        # options |= widgets.QFileDialog.DontUseNativeDialog
        default_name = self.current_file.replace("sorted", "curated")
        dir_name, base_name = os.path.split(default_name)
        default_save_path = os.path.join(dir_name, config.DEFAULT_SAVE_LOCATION, base_name)
        filename, _ = widgets.QFileDialog.getSaveFileName(
            self,
            "Save dataset",
            default_save_path,
            "(*.pkl)",
            options=options)

        if filename:
            self.save_dataset(filename)

    def load_dataset(self, filename):
        if filename.endswith("pkl"):
            dataset = suss.io.read_pickle(filename)
        elif filename.endswith("npy"):
            dataset = suss.io.read_numpy(filename)

        self.title = "SUSS Viewer - {}".format(filename)
        self.setWindowTitle(self.title)
        # After dataset is loaded, connect the save function
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
    """Splash screen displaying initial options"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = widgets.QVBoxLayout(self)
        self.main_button = widgets.QPushButton("Load Dataset", self)
        self.quit_button = widgets.QPushButton("Quit", self)
        layout.addWidget(self.main_button)
        layout.addWidget(self.quit_button)
        self.setLayout(layout)
        self.setMinimumSize(300, 100)


class SussViewer(widgets.QFrame):
    """Main window for working with a dataset

    Responsible for passing signals between the app components
    regarding changes to the dataset, as well as holding the app state.
    All subcomponents should reference the SussViewer object
    for the current dataset state, the currently selected clusters,
    and the currently highlighted cluster.
    """

    minimum_size = (1000, 500)
    animation_period = 100.0

    # Emits the new dataset object and the old dataset object
    UPDATED_CLUSTERS = pyqtSignal(object, object)
    # Emits a set of cluster labels that are currently selected and previously
    CLUSTER_SELECT = pyqtSignal(set, set)
    # Emits an integer label for the cluster that should be highlighted and previous
    CLUSTER_HIGHLIGHT = pyqtSignal(object, object, bool)
    AUDITORY_RESPONSES = pyqtSignal(bool)

    def __init__(self, dataset=None, parent=None):
        super().__init__(parent)
        self.stack = [("load", dataset)]
        self.redo_stack = []
        self.hidden = []

        self.selected = set()
        self._highlights_disabled = False
        self.highlighted = None

        self.animation_timer = QTimer()
        self.animation_timer.start(self.animation_period)

        main_menu = self.parent().menuBar()
        self.edit_menu = main_menu.addMenu("&Edit")
        self.tools_menu = main_menu.addMenu("&Tools")
        self.history_menu = main_menu.addMenu("&History")

        self.init_actions()
        self.init_ui()
        self.setup_shortcuts()

        self.UPDATED_CLUSTERS.connect(self.on_dataset_changed)
        self.parent().CLOSING_DATASET.connect(self.on_close_dataset)

    def on_close_dataset(self):
        main_menu = self.parent().menuBar()
        main_menu.removeAction(self.edit_menu.menuAction())
        main_menu.removeAction(self.history_menu.menuAction())

    def update_menu_bar(self):
        self.history_menu.clear()
        for act, dataset in reversed(self.stack):
            new_action = widgets.QAction("{} (n={})".format(act, len(dataset)), self)
            new_action.triggered.connect(partial(self.restore, dataset))
            self.history_menu.addAction(new_action)

    def init_actions(self):
        self.undo_action = widgets.QAction("Undo", self)
        self.redo_action = widgets.QAction("Redo", self)
        self.unhide_all_action = widgets.QAction("Reveal All", self)
        self.merge_action = widgets.QAction("Merge", self)
        self.delete_action = widgets.QAction("Delete", self)
        self.clear_action = widgets.QAction("Clear Selection", self)
        self.delete_others_action = widgets.QAction("Filter", self)
        self.select_all_action = widgets.QAction("Select All", self)
        self.show_auditory_action = widgets.QAction("Compute Sound Triggered PSTH", self, checkable=True)
        self.show_auditory_action.setChecked(False)

        self.unhide_all_action.triggered.connect(self.unhide_all)
        self.undo_action.triggered.connect(self._undo)
        self.redo_action.triggered.connect(self._redo)
        self.merge_action.triggered.connect(self.merge)
        self.delete_action.triggered.connect(self.delete)
        self.delete_others_action.triggered.connect(self.delete_unselected)
        self.clear_action.triggered.connect(self.clear)
        self.select_all_action.triggered.connect(self.select_all)
        self.show_auditory_action.triggered.connect(self.toggle_auditory)

    def toggle_auditory(self, state):
        self.AUDITORY_RESPONSES.emit(state)

    @contextmanager
    def temporary_highlight(self, label):
        prev_highlight = self.highlighted
        self.set_highlight(label, temporary=True)
        yield
        if prev_highlight in self.dataset.labels:
            self.set_highlight(prev_highlight, temporary=True)
        else:
            self.set_highlight(None, temporary=True)

    @contextmanager
    def disable_highlighting(self):
        """Use this to avoid inadvertent highlighting while operations are occuring"""
        self._highlights_disabled = True
        yield
        self._highlights_disabled = False

    def show_right_click_menu(self, label, pos, other_menus=tuple()):
        with self.temporary_highlight(label):
            menu = widgets.QMenu()

            if len(self.selected) >= 2:
                sel = list(self.selected)
                _merge_action = widgets.QAction(
                        "Merge Clusters {} and {}".format(
                            ", ".join(map(str, sel[:-1])), sel[-1]
                        ),
                        self
                )
                menu.addAction(_merge_action)
                _merge_action.triggered.connect(self.merge)

            if len(self.dataset.select(self.dataset.labels == label).flatten(1)) > 10:
                _recluster_action = widgets.QAction("Recluster Cluster {}".format(label), self)
                menu.addAction(_recluster_action)
                _recluster_action.triggered.connect(partial(self.recluster, label))

            _hide_action = widgets.QAction("Hide Cluster {}".format(label), self)
            menu.addAction(_hide_action)
            _hide_action.triggered.connect(partial(self.hide, label))

            menu.addSeparator()

            _delete_action = widgets.QAction("Delete Cluster {}".format(label), self)
            menu.addAction(_delete_action)
            _delete_action.triggered.connect(partial(self._delete, [label]))

            if len(self.selected) >= 2:
                _delete_all_action = widgets.QAction("Delete All Selected", self)
                menu.addAction(_delete_all_action)
                _delete_all_action.triggered.connect(self.delete)

            menu.addSeparator()

            if len(self.stack) > 1:
                menu.addAction(self.undo_action)

            if len(self.redo_stack) > 1:
                menu.addAction(self.redo_action)

            menu.addSeparator()

            for other_menu in other_menus:
                menu.addMenu(other_menu)

            menu.exec_(pos)

    def recluster(self, label):
        _new_dataset = recluster_node(self.dataset, label=label)
        new_labels = set(_new_dataset.labels)
        changed_labels = get_changed_labels(_new_dataset, self.dataset)

        _old_selected = self.selected.copy()
        self.selected = set.intersection(new_labels, changed_labels)
        self.colors = make_color_map(_new_dataset.labels)
        self._enstack("recluster", _new_dataset)
        self.CLUSTER_SELECT.emit(self.selected, _old_selected)

    @property
    def dataset(self):
        return self.stack[-1][1]

    @property
    def last_action(self):
        return self.stack[-1][0]

    def set_highlight(self, label, temporary=False):
        """Update highlight state and emit signal"""
        if self._highlights_disabled:
            return

        _old_label = self.highlighted
        if label == _old_label:
            return
        self.highlighted = label
        self.CLUSTER_HIGHLIGHT.emit(
                self.highlighted,
                _old_label if _old_label is not None else None,
                temporary)

    def set_selected(self, selected):
        """Update selection state and emit signal if changed"""
        _old_selected = self.selected.copy()
        if selected != self.selected:
            self.selected = selected
            self.CLUSTER_SELECT.emit(self.selected, _old_selected)

    def toggle_selected(self, label, selected):
        """Update selection state by setting a single label's state

        Emit signal only if the selection has changed"""
        _old_selected = self.selected.copy()
        if selected and label not in self.selected:
            self.selected.add(label)
            self.CLUSTER_SELECT.emit(self.selected, _old_selected)
        elif not selected and label in self.selected:
            self.selected.remove(label)
            self.CLUSTER_SELECT.emit(self.selected, _old_selected)
        else:
            pass

    def on_dataset_changed(self):
        self.colors = make_color_map(self.dataset.labels)
        self.update_menu_bar()

    @contextmanager
    def timer_paused(self):
        self.animation_timer.stop()
        yield
        self.animation_timer.start(self.animation_period)

    def setup_shortcuts(self):
        self.undo_action.setShortcut(gui.QKeySequence.Undo)
        self.window().addAction(self.undo_action)

        self.redo_action.setShortcut(gui.QKeySequence.Redo)
        self.window().addAction(self.redo_action)

        self.delete_action.setShortcut(gui.QKeySequence("Backspace"))
        self.window().addAction(self.delete_action)

        self.delete_others_action.setShortcut(gui.QKeySequence("Shift+Backspace"))
        self.window().addAction(self.delete_others_action)

        self.select_all_action.setShortcut(gui.QKeySequence.SelectAll)
        self.window().addAction(self.select_all_action)

        self.clear_action.setShortcut(gui.QKeySequence.Copy)
        self.window().addAction(self.clear_action)

        self.merge_action.setShortcut(gui.QKeySequence("Ctrl+M"))
        self.window().addAction(self.merge_action)

    def _enstack(self, action, dataset, clear_redo=True):
        with self.timer_paused():
            _old_dataset = self.dataset
            with self.disable_highlighting():
                if clear_redo:
                    self.redo_stack = []
                self.stack.append((action, dataset))
                self.colors = make_color_map(self.dataset.labels)
                self.UPDATED_CLUSTERS.emit(
                        self.dataset,
                        _old_dataset,
                )

    def _redo(self):
        if len(self.redo_stack) == 0:
            print("Nothing left to redo.")
            return

        last_action, _old_dataset = self.redo_stack.pop()
        self._enstack(last_action, _old_dataset, clear_redo=False)

    def _undo(self):
        if len(self.stack) == 1:
            print("Nothing left to undo.")
            return

        with self.timer_paused():
            last_action, _old_dataset = self.stack.pop()
            self.redo_stack.append((last_action, _old_dataset))

            new_labels = set(self.dataset.labels)
            changed_labels = get_changed_labels(self.dataset, _old_dataset)

            _old_selected = self.selected.copy()
            if "delete unselected" in last_action:
                self.selected = set(_old_dataset.labels)
            else:
                self.selected = set.intersection(new_labels, changed_labels)
            self.colors = make_color_map(self.dataset.labels)
            self.UPDATED_CLUSTERS.emit(
                self.dataset,
                _old_dataset
            )
            self.CLUSTER_SELECT.emit(self.selected, _old_selected)

    def restore(self, dataset):
        self._enstack("restored previous state", dataset)

    def reset(self):
        with self.timer_paused():
            _old_dataset = self.dataset
            self.stack = self.stack[:1]
            self.colors = make_color_map(self.dataset.labels)
            self.UPDATED_CLUSTERS.emit(
                self.dataset,
                _old_dataset
            )
            self.dataset_updated()

    def select_all(self):
        _old_selected = self.selected.copy()
        if self.selected == set(self.dataset.labels):
            self.selected = set()
        else:
            self.selected = set(self.dataset.labels)
        self.CLUSTER_SELECT.emit(self.selected, _old_selected)

    def hide(self, label):
        selector = match_one(self.dataset, label=label)
        node = self.dataset.nodes[selector][0]
        self.hidden.append(node)

        self._delete([label], action="hide")

    def unhide_all(self):
        _new_dataset = add_nodes(self.dataset, *self.hidden)
        self.hidden = []
        self._enstack("unhide", _new_dataset)

    def merge(self):
        if len(self.selected) < 2:
            widgets.QMessageBox.warning(
                    self,
                    "Merge failed",
                    "Not enough clusters selected to merge")
            return

        _new_dataset = merge_nodes(self.dataset, labels=self.selected)
        new_labels = set(_new_dataset.labels)
        changed_labels = get_changed_labels(_new_dataset, self.dataset)

        _old_selected = self.selected.copy()
        self.selected = set.intersection(new_labels, changed_labels)
        self.colors = make_color_map(_new_dataset.labels)
        self._enstack("merge", _new_dataset)
        self.CLUSTER_SELECT.emit(self.selected, _old_selected)

    def _delete(self, to_delete, action="delete"):
        if len(to_delete) == 0:
            widgets.QMessageBox.warning(
                    self,
                    "Delete failed",
                    "No clusters selected for deletion")
            return

        _new_dataset = delete_nodes(self.dataset, labels=list(to_delete))

        plural = "s" if len(to_delete) > 1 else ""
        self._enstack("{} node".format(action) + plural, _new_dataset)

    def delete(self):
        self._delete(self.selected)
        self.selected = set()

    def delete_unselected(self):
        labels = set(self.dataset.labels)
        to_delete = labels - self.selected
        _old_selected = self.selected.copy()
        self.selected = set()
        self._delete(to_delete, action="delete unselected")
        self.CLUSTER_SELECT.emit(self.selected, _old_selected)

    def save(self):
        self.parent().run_file_saver()

    def load(self):
        self.parent().run_file_loader()

    def clear(self):
        _old_selected = self.selected.copy()
        self.selected = set()
        self.CLUSTER_SELECT.emit(self.selected, _old_selected)
        self.CLUSTER_HIGHLIGHT.emit(None, self.highlighted, False)

    def init_ui(self):
        self.edit_menu.addAction(self.undo_action)
        self.edit_menu.addAction(self.redo_action)
        self.edit_menu.addSeparator()
        self.edit_menu.addAction(self.unhide_all_action)
        self.edit_menu.addAction(self.select_all_action)
        self.edit_menu.addAction(self.clear_action)
        self.edit_menu.addSeparator()
        self.edit_menu.addAction(self.merge_action)
        self.edit_menu.addAction(self.delete_action)
        self.edit_menu.addAction(self.delete_others_action)

        self.tools_menu.addAction(self.show_auditory_action)

        self.on_dataset_changed()
        # self.cluster_selector = ClusterSelector(parent=self)

        layout = widgets.QGridLayout()
        layout.setRowStretch(1, 1.5)
        layout.setRowStretch(2, 1.5)
        layout.setRowStretch(3, 2)
        layout.setColumnStretch(1, 1.5)
        layout.setColumnStretch(2, 1.5)
        layout.setColumnStretch(3, 1)

        # Initialize TSNEPlot first so that the T-SNE embedding
        # can be computed in the background while the other components
        # are being initialized

        # row, col, rowspan, colspan
        layout.addWidget(TSNEPlot(parent=self), 1, 1, 2, 1)
        layout.addWidget(ProjectionsPlot(parent=self), 1, 2, 2, 1)
        layout.addWidget(ClusterSelector(parent=self), 1, 0, 3, 1)
        layout.addWidget(WaveformsPlot(parent=self), 2, 3, 1, 1)
        layout.addWidget(ISIPlot(parent=self), 1, 3, 1, 1)
        layout.addWidget(TimeseriesPlot(parent=self), 3, 1, 1, 3)

        self.setLayout(layout)
        self.setMinimumSize(*self.minimum_size)


if __name__ == "__main__":
    app = widgets.QApplication(sys.argv)
    window = App()
    try:
        result = app.exec_()
    except:
        recovery_file = "{}.recovery.pkl".format(
                os.path.basename(window.current_file))
        print(
            "A horrible error has occured. "
            "Saving recovery file at {}".format(recovery_file))
        window.save_dataset(recovery_file)
        raise
    sys.exit(result)
