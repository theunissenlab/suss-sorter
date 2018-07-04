import numpy as np
from PyQt5 import QtWidgets as widgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class WaveformsPlot(widgets.QFrame):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setup_plots()
        self.setup_data()
        self.init_ui()
        self.parent().UPDATED_CLUSTERS.connect(self.reset)
        self.parent().CLUSTER_HIGHLIGHT.connect(self.on_cluster_highlight)
        self.parent().CLUSTER_SELECT.connect(self.on_cluster_select)

    def reset(self, new_dataset, old_dataset):
        self.ax.clear()
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
        fig = Figure()
        fig.patch.set_alpha(0.0)
        self.canvas = FigureCanvas(fig)
        self.canvas.setStyleSheet("background-color:transparent;")

        self.ax = fig.add_axes(
                [0, 0, 1, 1],
                facecolor="#111111")
        self.ax.patch.set_alpha(0.8)
        self.highlight_plot = None

    def setup_data(self):
        if not len(self.dataset.nodes):
            self.canvas.draw_idle()
            return

    def on_cluster_select(self, selected, old_selected):
        if None in selected:
            print("SOMNWHOW NONE GOT IN SELECTED")

        self.ax.clear()
        self.ax.patch.set_alpha(0.8)
        self.highlight_plot = None

        for label in selected:
            if label is None:
                continue
            try:
                node = self.dataset.nodes[self.dataset.labels == label][0]
            except:
                # FIXME (kevin): haven't fiugred out this bug...
                # but don't want it to crash
                print("An error occured")
                print("Couldnt find", label, "in", self.dataset.labels)
                continue
            mean = node.centroid
            std = np.std(node.waveforms, axis=0)

            self.ax.fill_between(
                np.arange(len(mean)),
                mean - std,
                mean + std,
                color=self.colors[label],
                alpha=0.2,
                rasterized=True)
            self.ax.plot(
                np.arange(len(mean)),
                mean,
                color=self.colors[label],
                linewidth=3,
                rasterized=True)
        self.canvas.draw_idle()

    def on_cluster_highlight(self, new_highlight, old_highlight):
        if new_highlight is None:
            if self.highlight_plot is not None:
                self.highlight_plot.set_visible(False)
            return

        node = self.dataset.nodes[self.dataset.labels == new_highlight][0]
        mean = node.centroid
        # std = np.std(node.waveforms, axis=0)

        if new_highlight is not None and self.highlight_plot is None:
            self.highlight_plot, = self.ax.plot(
                np.arange(len(mean)),
                mean,
                color=self.colors[new_highlight],
                alpha=1.0,
                linewidth=2,
                linestyle="--"
            )
        elif new_highlight is not None and self.highlight_plot is not None:
            self.highlight_plot.set_color(self.colors[new_highlight])
            self.highlight_plot.set_ydata(mean)
            self.highlight_plot.set_visible(True)

        self.highlight_plot.set_visible(True)
        self.canvas.draw_idle()

    def init_ui(self):
        layout = widgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
