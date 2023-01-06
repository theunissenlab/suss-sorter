import numpy as np
from PyQt5 import QtWidgets as widgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class WaveformsPlot(widgets.QFrame):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view_all = False
        self.show_max = 30

        self.setup_plots()
        self.setup_data()
        self.init_ui()
        self.parent().UPDATED_CLUSTERS.connect(self.reset)
        self.parent().CLUSTER_HIGHLIGHT.connect(self.on_cluster_highlight)
        self.parent().CLUSTER_SELECT.connect(self.on_cluster_select)

    def toggle_view_all_waveforms(self, state):
        if state != self.view_all:
            self.view_all = state
            self.reset(self.dataset, self.dataset)

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
        self.ax.clear()
        self.ax.patch.set_alpha(0.8)
        self.highlight_plot = None

        if not len(self.dataset.nodes):
            self.canvas.draw_idle()
            return

        for label in self.selected:
            if label is None:
                continue
            try:
                node = self.dataset.nodes[self.dataset.labels == label][0]
            except:
                continue

            mean = node.centroid
            std = np.std(node.flatten().waveforms, axis=0)
            if self.view_all:
                skip = max(1, node.count // self.show_max)
                start_at = np.random.choice(np.arange(skip))
                self.ax.plot(
                    node.flatten().waveforms[start_at::skip].T,
                    color=self.colors[label],
                    rasterized=True,
                    alpha=0.5,
                    linewidth=0.5)
            else:
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
                linewidth=2,
                rasterized=True)
            self.ax.set_ylim(
                min(self.ax.get_ylim()[0], np.min(mean - std)),
                max(self.ax.get_ylim()[1], np.max(mean + std))
            )
        self.canvas.draw_idle()

    def on_cluster_select(self, selected, old_selected):
        self.setup_data()

    def on_cluster_highlight(self, new_highlight, old_highlight, temporary):
        if new_highlight is None:
            if self.highlight_plot is not None:
                self.highlight_plot.set_visible(False)
            self.canvas.draw_idle()
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

        toggle_box = widgets.QCheckBox("View All Waveforms",self)
        toggle_box.stateChanged.connect(self.toggle_view_all_waveforms)
        layout.addWidget(toggle_box)

        self.setLayout(layout)
