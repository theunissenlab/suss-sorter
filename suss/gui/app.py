import numpy as np
from PyQt5 import QtWidgets as widgets

from components import (ProjectionsPane,
        OverviewScatterPane,
        TimeseriesPane,
        WaveformsPane,
        ISIPane,
        get_color_dict,
        selector_area
        )


class SussViewer(widgets.QFrame):

    def __init__(self, dataset, parent=None):
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

        scroll_area = selector_area(self.dataset, 140, color_dict, self.toggle)

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
        outer_2.addWidget(scroll_area)
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
    import sys
    import suss.io

    app = widgets.QApplication(sys.argv)

    dataset = suss.io.read_pickle(sys.argv[1])
    main = widgets.QMainWindow()
    viewer = SussViewer(dataset)
    main.setCentralWidget(viewer)
    main.show()
    sys.exit(app.exec_())

