import numpy as np
from PyQt5 import QtWidgets as widgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

from suss.gui.utils import clear_axes


class ProjectionsPlot(widgets.QFrame):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_plots()
        self.setup_data()
        self.init_ui()
        self.parent().UPDATED_CLUSTERS.connect(self.reset)
        self.parent().CLUSTER_HIGHLIGHT.connect(self.on_cluster_highlight)
        self.parent().CLUSTER_SELECT.connect(self.on_cluster_select)

    def reset(self, new_dataset, old_dataset):
        self.ax_1d.clear()
        self.ax_2d.clear()
        self.canvas.draw_idle()
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
        fig = Figure(facecolor="#C0C0C0")
        fig.patch.set_alpha(1.0)
        fig.set_rasterized(True)
        self.canvas = FigureCanvas(fig)
        self.canvas.setStyleSheet("background-color:transparent;")

        self.ax_2d = fig.add_axes(
                [0, 0.25, 1, 0.75],
                facecolor="#111111")
        self.ax_2d.patch.set_alpha(0.8)
        self.ax_1d = fig.add_axes(
                [0, 0, 1, 0.25],
                facecolor="#111111")
        self.ax_1d.patch.set_alpha(0.8)
        clear_axes(self.ax_1d, self.ax_2d)

    def setup_data(self):
        if not len(self.dataset.nodes):
            self.canvas.draw_idle()
            return

    def on_cluster_select(self, selected, old_selected):
        self.ax_1d.clear()
        self.ax_2d.clear()
        clear_axes(self.ax_1d, self.ax_2d)
        self.ax_1d.patch.set_alpha(0.8)
        self.ax_2d.patch.set_alpha(0.8)

        if not len(selected):
            self.canvas.draw_idle()
            return

        selected_data = self.dataset.select(
            np.isin(self.dataset.labels, list(selected))
        )

        if not len(selected_data.flatten(1)):
            self.canvas.draw_idle()
            return

        skip = max(1, int(selected_data.count / 1000))
        self.projector = PCA(n_components=2).fit(selected_data.flatten(1).waveforms)

        projected = [
            self.projector.transform(node.flatten().waveforms[::1 if skip > len(node.flatten().waveforms) else skip])
            for node in selected_data.nodes
        ]

        for label, data in zip(selected_data.labels, projected):
            self.ax_2d.scatter(
                *data.T[:2],
                color=self.colors[label],
                s=1,
                alpha=1,
                rasterized=True
            )

        self.ax_1d.hist(
                [data[:, 0] for data in projected],
                bins=100,
                color=[self.colors[label] for label in selected_data.labels],
                alpha=0.9,
                stacked=True,
                rasterized=True
        )
        self.canvas.draw_idle()

    def on_cluster_highlight(self, new_highlight, old_highlight, temporary):
        pass

    def init_ui(self):
        layout = widgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
