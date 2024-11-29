import numpy as np
from derotation.simulate.basic_rotator import Rotator
from matplotlib import pyplot as plt

from test_finding_center_of_rotation_by_joining_two_pipelines import (
    create_sample_image_with_two_cells,
    create_image_stack, 
    create_rotation_angles,
    MockIncrementalPipeline
)

plane_angle = 5


#  create a sample image with two cells
cells = create_sample_image_with_two_cells()
#  chnage all zeros 
cells[cells == 0] = 80

num_frames = 15
image_stack = create_image_stack(cells, num_frames=num_frames)

angles, _ = create_rotation_angles(image_stack.shape)

#  rotate the image stack
rotator = Rotator(angles, image_stack, rotation_plane_angle=plane_angle)
rotated_image_stack = rotator.rotate_by_line()


#  plot the original image, the rotated images 
fig, ax = plt.subplots(2, num_frames // 2 + 2, figsize=(25, 10))

print(ax.shape)
ax[0, 0].imshow(image_stack[0], cmap="gray")
ax[0, 0].set_title("Original image")
ax[0, 0].axis("off")

for n in range(len(rotated_image_stack) + 1):
    row = n // 2 + 1
    col = n % 2
    ax[row, col].imshow(rotated_image_stack[n], cmap="gray")
    ax[row, col].set_title(f"Rotated image {n}")
    ax[row, col].axis("off")

    # add information about the rotation angle
    angles = rotator.angles[n]
    angle_range = f"angles: {angles.min():.0f}-{angles.max():.0f}"
    ax[row, col + 1].text(
        0.5,
        0.9,
        angle_range,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax[row, col + 1].transAxes,
        color="white",
    )

plt.show()


#  fit an ellipse to the centers of the largest blobs in each image
DI = MockIncrementalPipeline(
    rotated_image_stack, angles
)

coord_first_blob_of_every_image = DI.get_coords_of_largest_blob(
        rotated_image_stack
    )

# Fit an ellipse to the largest blob centers and get its center
center_x, center_y, a, b, theta = DI.fit_ellipse_to_points(
    coord_first_blob_of_every_image
)

DI.plot_ellipse_fit_and_centers(
    coord_first_blob_of_every_image,
    center_x,
    center_y,
    a,
    b,
    theta,
    rotated_image_stack,
    num_frames,
)