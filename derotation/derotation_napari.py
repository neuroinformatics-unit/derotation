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
            self.pipeline.image_stack, name="image", colormap="turbo"
        )
        self.setLayout(QVBoxLayout())

        self.analyze_button = QPushButton()
        self.analyze_button.setText("Run analysis")
        self.analyze_button.clicked.connect(self.analog_data_analysis)
        self.layout().addWidget(self.analyze_button)

        self.rotate_by_line_button = QPushButton()
        self.rotate_by_line_button.setText("Rotate by line")
        self.rotate_by_line_button.clicked.connect(
            self.rotate_images_using_line_clock
        )
        self.layout().addWidget(self.rotate_by_line_button)

        self.save_button = QPushButton()
        self.save_button.setText("Save")
        self.save_button.clicked.connect(self.save)
        self.layout().addWidget(self.save_button)

        self.mpl_widget = DerotationCanvas(self._viewer)
        self.layout().addWidget(self.mpl_widget)

    def analog_data_analysis(self):
        self.pipeline.process_analog_signals()

        self.mpl_widget.angles_over_time = self.pipeline.rot_deg_line

        self.mpl_widget.draw()
        print("Data analysis done")

    def rotate_images_using_line_clock(self):
        self.rotated_images = self.pipeline.rotate_frames_line_by_line()
        self._viewer.add_image(
            np.array(self.rotated_images),
            name="rotated_images",
            colormap="turbo",
        )
        print("Images rotated")

    def save(self):
        self.masked_img_array = self.pipeline.add_circle_mask()

        self._viewer.add_image(
            self.masked_img_array, name="masked", colormap="turbo"
        )
        print("Images masked")

        self.pipeline.save(self.masked_img_array)

        print("Images saved")
