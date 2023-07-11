from typing import Optional

import numpy as np
from napari.viewer import Viewer
from napari_matplotlib.base import SingleAxesWidget
from qtpy.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from derotation.analysis.derotation_pipeline import DerotationPipeline


class DerotationCanvas(SingleAxesWidget):
    def __init__(
        self,
        napari_viewer: Viewer,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(napari_viewer, parent=parent)
        self.angles_over_time = np.zeros(100)
        self._update_layers(None)

    def draw(self):
        self.axes.clear()
        self.axes.plot(self.angles_over_time, color="red")
        self.axes.set_title(f"z={self.current_z}")
        self.axes.axvline(self.current_z)


class Plotting(QWidget):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()

        self._viewer = napari_viewer

        self.pipeline = DerotationPipeline()
        self._viewer.add_image(
            self.pipeline.image, name="image", colormap="turbo"
        )
        self.setLayout(QVBoxLayout())

        self.analyze_button = QPushButton()
        self.analyze_button.setText("Run analysis")
        self.analyze_button.clicked.connect(self.analysis)
        self.layout().addWidget(self.analyze_button)

        self.mpl_widget = DerotationCanvas(self._viewer)
        self.layout().addWidget(self.mpl_widget)

    def analysis(self):
        self.pipeline.process_analog_signals()

        self.mpl_widget.angles_over_time = (
            self.pipeline.image_rotation_degree_per_frame
        )

        self.mpl_widget.draw()

        self.pipeline.get_clean_centroids()

        centers = [
            [t, coord[0], coord[1]]
            for t, coord in enumerate(self.pipeline.correct_centers)
        ]

        self._viewer.add_points(
            centers,
        )
