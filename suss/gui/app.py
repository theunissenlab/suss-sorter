import sys

import numpy as np
from PyQt5 import QtWidgets as widgets

import suss.io
from components import (
    ClusterSelector,
    ISIPane,
    OverviewScatterPane,
    ProjectionsPane,
    TimeseriesPane,
    WaveformsPane,
    get_color_dict,
    selector_area
)


class App(widgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = "SUSS Viewer"
        self.suss_viewer = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("File")
        load_action = widgets.QAction("Find dataset", self)
        load_action = widgets.QAction("Close", self)
        fileMenu.addAction(load_action)
        load_action.triggered.connect(self.load_dataset)
        load_action.triggered.connect(self.close)

        self.dataset_loader = DatasetLoader(self)
        self.setCentralWidget(self.dataset_loader)
        self.dataset_loader.main_button.clicked.connect(self.load_dataset)
        self.dataset_loader.quit_button.clicked.connect(self.close)

        rect = self.frameGeometry()
        center = widgets.QDesktopWidget().availableGeometry().center()
        rect.moveCenter(center)
        self.move(rect.topLeft())

        self.show()
            
    def load_dataset(self):
        options = widgets.QFileDialog.Options()
        options |= widgets.QFileDialog.DontUseNativeDialog
        selected_file, _ = widgets.QFileDialog.getOpenFileName(self,
            "Pickle Files (*.pkl)",
            options=options)
        if not selected_file:
            return
        else:
            dataset = suss.io.read_pickle(selected_file)
            self.suss_viewer = SussViewer(dataset, self)
            self.setCentralWidget(self.suss_viewer)
            self.show()


class DatasetLoader(widgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = widgets.QVBoxLayout(self)
        self.main_button = widgets.QPushButton("Load Dataset", self)
        self.quit_button = widgets.QPushButton("Quit", self)
        layout.addWidget(self.main_button)
        layout.addWidget(self.quit_button)
        self.setLayout(layout)


class SussViewer(widgets.QFrame):

    def __init__(self, dataset=None, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.active_clusters = set()
        self.setup_panes()

    def setup_panes(self):
        color_dict = get_color_dict(self.dataset.labels)
        self.projections = ProjectionsPane(self.dataset, color_dict, size=(300, 350), facecolor="#666666")
        self.overview = OverviewScatterPane(self.dataset, color_dict, size=(100, 100), facecolor="#444444")
        self.timeseries = TimeseriesPane(self.dataset, color_dict, size=(700, 100), facecolor="#888888")
        self.waveforms = WaveformsPane(self.dataset, color_dict, size=(300, 250), facecolor="#444444")
        self.isi = ISIPane(self.dataset, color_dict, size=(200, 100), facecolor="#444444")
        self.cluster_selector = ClusterSelector(self.dataset, color_dict, self.toggle)

        # scroll_area = selector_area(self.dataset, 140, color_dict, self.toggle)

        outer_2 = widgets.QHBoxLayout()
        outer_1 = widgets.QVBoxLayout()
        inner_1 = widgets.QHBoxLayout()
        inner_2 = widgets.QVBoxLayout()
        inner_3 = widgets.QHBoxLayout()
        inner_1.addWidget(self.projections)
        inner_3.addWidget(self.isi)
        inner_3.addWidget(self.overview)
        inner_2.addLayout(inner_3)
        inner_2.addWidget(self.waveforms)
        inner_1.addLayout(inner_2)
        outer_1.addLayout(inner_1)
        outer_1.addWidget(self.timeseries)
        outer_2.addWidget(self.cluster_selector)
        outer_2.addLayout(outer_1)

        self.setLayout(outer_2)

    def toggle(self, selected, label=None):
        if selected:
            self.active_clusters.add(label)
        else:
            self.active_clusters.remove(label)
        self.projections.toggle(selected, label)
        self.overview.toggle(selected, label)
        self.timeseries.toggle(selected, label)
        self.waveforms.toggle(selected, label)
        self.isi.toggle(selected, label)


if __name__ == "__main__":
    app = widgets.QApplication(sys.argv)
    window = App()
    sys.exit(app.exec_())

