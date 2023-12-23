from typing import Optional

import numpy as np
from napari.viewer import Viewer
from napari_matplotlib.base import SingleAxesWidget
from qtpy.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class DerotationCanvas(SingleAxesWidget):
    def __init__(
        self,
        napari_viewer: Viewer,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(napari_viewer, parent=parent)
        self.angles_over_time = np.sin(np.linspace(0, 10, 100))
        self._update_layers(None)

    def draw(self):
        self.axes.plot(self.angles_over_time, color="red")
        self.axes.set_title(f"z={self.current_z}")
        self.axes.axvline(self.current_z)


class Plotting(QWidget):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()

        self._viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.button = QPushButton()
        self.button.setText("Make plot")
        self.layout().addWidget(self.button)

        self.mpl_widget = DerotationCanvas(self._viewer)
        self.layout().addWidget(self.mpl_widget)
