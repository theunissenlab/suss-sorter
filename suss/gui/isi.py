import numpy as np
from PyQt5 import QtWidgets as widgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import suss.gui.config as config


class ISIPlot(widgets.QFrame):

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
        fig = Figure(facecolor="#C0C0C0")
        fig.patch.set_alpha(1.0)
        self.canvas = FigureCanvas(fig)
        self.canvas.setStyleSheet("background-color:transparent;")

        self.ax = fig.add_axes(
                [0, 0.15, 1, 0.85],
                facecolor="#C0C0C0")
        self.ax.patch.set_alpha(0.8)
        self.ax.set_xlim(0, config.ISI_MAX)
        self.ax.set_xticks([0.001, 0.02])
        self.ax.set_xticklabels(
                ["1ms", "20ms"],
                horizontalalignment="center",
                fontsize=5)
        for tick in self.ax.get_xaxis().get_major_ticks():
            tick.set_pad(0)

        self.text_ax = fig.add_axes(
                [0, 0, 1, 1],
                xlim=(0, 1),
                ylim=(0, 1))
        self.text_ax.patch.set_alpha(0.0)
        self.isi_label = self.text_ax.text(
            0.98,
            0.95,
            "",
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=8,
            color="White")

    def setup_data(self):
        if not len(self.dataset.nodes):
            self.canvas.draw_idle()
            return

    def on_cluster_select(self, selected, old_selected):
        if not len(selected):
            self.isi_label.set_text("")
        self.ax.clear()
        self.ax.set_xlim(0, config.ISI_MAX)
        self.ax.set_xticks([0.001, 0.02] + ([] if config.ISI_MAX <= 0.1 else [0.1]))
        self.ax.set_xticklabels(
                ["1ms", "20ms"] + ([] if config.ISI_MAX <= 0.1 else ["100ms"]),
                horizontalalignment="left",
                fontsize=5)
        for tick in self.ax.get_xaxis().get_major_ticks():
            tick.set_pad(0)

        self.ax.patch.set_alpha(0.8)

        clusters = self.dataset.select(
            np.isin(self.dataset.labels, list(selected))
        ).flatten()

        isi = np.diff(clusters.times)
        if not len(isi):
            self.canvas.draw_idle()
            return

        isi_violations = len(np.where(isi < 0.001)[0]) / len(isi)

        across_clusters = clusters.labels[:-1] != clusters.labels[1:]
        within_cluster = clusters.labels[:-1] == clusters.labels[1:]

        if not np.sum(across_clusters) + np.sum(within_cluster):
            self.canvas.draw_idle()
            return

        fr = len(clusters) / (np.max(clusters.times) - np.min(clusters.times))

        self.ax.hist(
            [
                isi[across_clusters],
                isi[within_cluster]
            ],
            bins=config.ISI_BINS,
            density=True,
            range=(0, config.ISI_MAX),
            stacked=True,
            alpha=0.8,
            color=["Orange", "Black"]
        )
        self.ax.vlines(
                0.001,
                *self.ax.get_ylim(),
                color="Red",
                linestyle="--",
                linewidth=0.5)

        if len(isi[across_clusters]):
            isi_violations_across = (
                    len(np.where(isi[across_clusters] < 0.001)[0]) /
                    len(isi[across_clusters])
            )
            self.isi_label.set_text(
                    "{:.1f}% ISI violations\np = {:.1f}%\n{:.1f}% across clusters".format(
                    100.0 * isi_violations,
                    100 * (1 - np.exp(-fr * 0.001)),
                    100.0 * isi_violations_across,
                )
            )
        else:
            self.isi_label.set_text(
                "{:.1f}% ISI violations\np = {:.1f}%".format(
                    100.0 * isi_violations,
                    100 * (1 - np.exp(-fr * 0.001)),
                )
            )

        self.canvas.draw_idle()

    def on_cluster_highlight(self, new_highlight, old_highlight, temporary):
        pass

    def init_ui(self):
        layout = widgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
