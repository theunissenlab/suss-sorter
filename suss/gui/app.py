import sys
import numpy as np

from functools import partial

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5 import QtWidgets as widgets
from PyQt5.QtWidgets import (
        QApplication,
        QDialog,
        QPushButton,
        QVBoxLayout,
        QHBoxLayout,
        QMainWindow,
)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from matplotlib import cm

class SussWindow(widgets.QDialog):

    def __init__(self, dataset, scatter_data, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.scatter_data = scatter_data
        self.setup_gui()

        self.labels = self.dataset.flatten(1, assign_labels=True).labels
        self.base_scatter = self.scatter_ax.scatter(
                *scatter_data.T,
                color="Gray",
                alpha=0.5,
                s=5)
        self.label_scatters = {}
        self.label_temporal_scatters = {}

        self.flattened_dataset = dataset.flatten(assign_labels=True)
        full_lda1 = LDA(n_components=1).fit_transform(self.flattened_dataset.waveforms, self.flattened_dataset.labels)

        self.label_colors = {}
        for label in np.unique(self.labels):
            self.label_scatters[label] = self.scatter_ax.scatter([], [], s=5)
            self.label_colors[label] = self.label_scatters[label].get_edgecolor()[0]
            self.label_temporal_scatters[label] = self.temporal_ax.scatter(
                self.flattened_dataset.times[self.flattened_dataset.labels == label][::10],
                full_lda1[self.flattened_dataset.labels == label][::10],
                s=2,
                alpha=0.01,
                color="#dddddd"
            )

        self.selected_labels = set()

    def setup_gui(self):
        self.selector_fig = Figure()
        self.selector_canvas = FigureCanvas(self.selector_fig)
        self.scatter_ax = self.selector_fig.add_axes([0, 0.2, 0.6, 0.8])
        self.temporal_ax = self.selector_fig.add_axes([0, 0, 1, 0.2])
        self.lda_2d = self.selector_fig.add_axes([0.6, 0.5, 0.4, 0.5])
        self.lda_1d = self.selector_fig.add_axes([0.6, 0.2, 0.4, 0.3])
        self.selector_canvas.draw()

        button_layout = widgets.QHBoxLayout()
        for label, cluster in zip(self.dataset.labels, self.dataset.nodes):
            button = widgets.QPushButton("{}".format(label), self)
            button.setCheckable(True)
            button.setDefault(False)
            button.setAutoDefault(False)
            button.clicked[bool].connect(partial(self.on_click, label=label))
            button_layout.addWidget(button)

        main_layout = widgets.QVBoxLayout()
        main_layout.addWidget(self.selector_canvas)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def on_click(self, state, label=None):
        if state:
            self.label_scatters[label].set_offsets(
                self.scatter_data[self.labels == label]
            )
            self.label_temporal_scatters[label].set_alpha(0.2)
            self.label_temporal_scatters[label].set_color(self.label_colors[label])
            self.selected_labels.add(label)
        else:
            self.label_scatters[label].set_offsets(np.zeros((0, 2)))
            self.label_temporal_scatters[label].set_alpha(0.01)
            self.label_temporal_scatters[label].set_color("#dddddd")
            self.selected_labels.remove(label)

        self.update_lda()
        self.selector_canvas.draw()

    def update_lda(self):
        self.lda_2d.clear()
        if len(self.selected_labels) == 0:
            return

        selected_labels = list(self.selected_labels)
        selected_clusters = self.dataset.select(
                np.isin(self.dataset.labels, selected_labels)
        )

        selected_data = selected_clusters.flatten(assign_labels=True)

        if len(self.selected_labels) == 1:
            lda2 = PCA(n_components=2).fit(selected_data.waveforms, selected_data.labels)
            lda1 = PCA(n_components=1).fit(selected_data.waveforms, selected_data.labels)
        elif len(self.selected_labels) == 2:
            lda2 = PCA(n_components=2).fit(selected_data.waveforms, selected_data.labels)
            lda1 = LDA(n_components=1).fit(selected_data.waveforms, selected_data.labels)
        else:
            lda2 = LDA(n_components=2).fit(selected_data.waveforms, selected_data.labels)
            lda1 = LDA(n_components=1).fit(selected_data.waveforms, selected_data.labels)

        lda2_data = lda2.transform(selected_data.waveforms)
        lda1_data = lda1.transform(selected_data.waveforms)

        for label in self.selected_labels:
            self.lda_2d.scatter(*lda2_data[selected_data.labels == label][::10].T, s=1, color=self.label_colors.get(label), alpha=0.2)

        violation_idx = np.diff(selected_data.times) <= 0.001

        pairs = zip(lda2_data[:-1][::10][violation_idx[::10]], lda2_data[1:][::10][violation_idx[::10]])
        for pair in pairs:
            self.lda_2d.plot(*zip(pair[0], pair[1]), linewidth=0.5, color="#222222", alpha=1.0)

        self.lda_1d.clear()
        self.lda_1d.hist(lda1_data, bins=200, color="#444444", alpha=0.9)


if __name__ == "__main__":
    import suss.io

    app = widgets.QApplication(sys.argv)

    sorted_dataset = suss.io.read_pickle("/Users/kevinyu/Projects/solid-garbanzo/"
            "datasets/GreYel_sorted-e10.pkl")
    scatter_data = suss.io.read_numpy("/Users/kevinyu/Projects/solid-garbanzo/"
            "datasets/GreYel_spacetime-e10.npy")

    main = SussWindow(sorted_dataset, scatter_data)
    main.show()
    sys.exit(app.exec_())




'''


class Window(QDialog):

    def __init__(self, dataset, parent=None):
        super().__init__(parent)
        self.dataset = dataset

    def setup_gui(self):
        self.selector_fig = plt.figure()
        self.selector_canvas = FigureCanvas(self.selector_fig)

        self.main_layout = QHboxLayout

        # self.toolbar = NavigationToolbar(self.canvas, self)

        # self.button = QPushButton("Plot Now")
        # self.button.clicked.connect(self.plot)

        self.scatter_ax = self.figure.add_axes([0, 0, 1, 1])
        self.scatter_ax.axis("off")

        final_clusters = dataset.flatten(1, assign_labels=True)
        tsned_data = TSNE(n_components=2).fit_transform(final_clusters.waveforms)
        # for print(len(dataset.nodes))
        for label in np.unique(final_clusters.labels):
            self.scatter_ax.scatter(*tsned_data[final_clusters.labels == label].T)

        self.canvas.draw()

        layout = QHboxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        # layout.addWidget(self.button)
        self.setLayout(layout)
'''
