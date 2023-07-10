import matplotlib.pyplot as plt

from qtpy.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napari.viewer import Viewer



class Plotting(QWidget):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()

        self._viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.button = QPushButton()
        self.button.setText("Make plot")
        self.button.clicked.connect(
            self._make_plot
        )
        self.layout().addWidget(self.button)

        self._viewer.dims.events.connect(self._print_z)

    def _make_plot(self):
        plt.plot([1,2,3], [5,6,7])
        plt.show()

    def _print_z(self):
        z = self._viewer.dims.current_step[0]
        print(z)

        plt.clear()
        plt.axvline(z)
        plt.show()

     