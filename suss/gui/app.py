import sys
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QApplication, QDialog, QPushButton, QVBoxLayout

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class Window(QDialog):

    def __init__(self, dataset, parent=None):
        super().__init__(parent)
        self.dataset = dataset

        self.figure = plt.figure()

        self.canvas = FigureCanvas(self.figure)

        self.toolbar = NavigationToolbar(self.canvas, self)

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

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        # layout.addWidget(self.button)
        self.setLayout(layout)
