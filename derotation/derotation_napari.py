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
from derotation.analysis.find_centroid import detect_blobs, preprocess_image
from derotation.analysis.rigid_registration import refine_derotation
from derotation.analysis.rotate_images import image_stack_rotation


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
        self.analyze_button.clicked.connect(self.analog_data_analysis)
        self.layout().addWidget(self.analyze_button)

        self.find_centroids_button = QPushButton()
        self.find_centroids_button.setText("Find centroids")
        self.find_centroids_button.clicked.connect(self.find_centroids)
        self.layout().addWidget(self.find_centroids_button)

        self.rotate_images_button = QPushButton()
        self.rotate_images_button.setText("Rotate images")
        self.rotate_images_button.clicked.connect(
            self.rotate_images_using_motor_feedback
        )
        self.layout().addWidget(self.rotate_images_button)

        self.refine_derotation_button = QPushButton()
        self.refine_derotation_button.setText("Refine derotation")
        self.refine_derotation_button.clicked.connect(self.refine_derotation)
        self.layout().addWidget(self.refine_derotation_button)

        self.labeled_rotated_button = QPushButton()
        self.labeled_rotated_button.setText("Labeled rotated")
        self.labeled_rotated_button.clicked.connect(self.label_derotated)
        self.layout().addWidget(self.labeled_rotated_button)

        self.mpl_widget = DerotationCanvas(self._viewer)
        self.layout().addWidget(self.mpl_widget)

    def analog_data_analysis(self):
        self.pipeline.process_analog_signals()

        self.mpl_widget.angles_over_time = (
            self.pipeline.image_rotation_degree_per_frame
        )

        self.mpl_widget.draw()
        print("Data analysis done")

    def find_centroids(self):
        self.pipeline.get_clean_centroids()

        centers = [
            [t, coord[0], coord[1]]
            for t, coord in enumerate(self.pipeline.correct_centers)
        ]

        self._viewer.add_points(
            centers,
        )
        print("Centroids found")

    def rotate_images_using_motor_feedback(self):
        self.rotated_images = image_stack_rotation(
            self.pipeline.image, self.pipeline.image_rotation_degree_per_frame
        )
        self._viewer.add_image(
            np.array(self.rotated_images),
            name="rotated_images",
            colormap="turbo",
        )
        print("Images rotated")

    def mask_images(self):
        #  exclude borders of 50 pixels, make smaller array
        length = len(self.rotated_images[0]) - 100
        self.rotated_images_masked = np.zeros(
            (len(self.rotated_images), length, length)
        )
        for i, image in enumerate(self.rotated_images):
            self.rotated_images_masked[i] = image[50:-50, 50:-50]
        self.rotated_images_masked = [o for o in self.rotated_images_masked]

        self._viewer.add_image(
            np.array(self.rotated_images_masked),
            name="rotated_images_masked",
            colormap="turbo",
        )

    def refine_derotation(self):
        self.mask_images()

        output = refine_derotation(self.rotated_images_masked)
        refined_rotated_images = [o["timg"] for o in output]

        self._viewer.add_image(
            np.array(refined_rotated_images),
            name="refined_rotated_images",
            colormap="turbo",
        )
        print("Image rotation refined")

    def label_derotated(self):
        labels = self.label_derotated_images(self.rotated_images_masked)
        self._viewer.add_image(
            np.array(labels),
            name="derotated_labels",
            colormap="turbo",
        )

    @staticmethod
    def label_derotated_images(image_stack):
        labels = []
        for img in image_stack:
            img = preprocess_image(img)
            label, _ = detect_blobs(img)
            labels.append(label)
        return labels
