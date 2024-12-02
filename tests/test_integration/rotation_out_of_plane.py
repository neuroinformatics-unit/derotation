import numpy as np
from matplotlib import pyplot as plt
from test_finding_center_of_rotation_by_joining_two_pipelines import (
    create_image_stack,
    create_rotation_angles,
    create_sample_image_with_two_cells,
)

from derotation.simulate.basic_rotator import Rotator

plane_angle = 25
num_frames = 50
pad = 20

#  create a sample image with two cells
cells = create_sample_image_with_two_cells(lines_per_frame=100)
cells = np.pad(cells, ((pad, pad), (pad, pad)), mode="constant")
cells[cells == 0] = 80


image_stack = create_image_stack(cells, num_frames=num_frames)
print("Image stack shape:", image_stack.shape)

_, angles = create_rotation_angles(image_stack.shape)
print("Angles shape:", angles.shape)

#  rotate the image stack
rotator = Rotator(angles, image_stack, rotation_plane_angle=plane_angle)
rotated_image_stack = rotator.rotate_by_line()

print("Rotated image stack shape:", rotated_image_stack.shape)

#  plot the original image, the rotated images
row_n = 5
fig, ax = plt.subplots(row_n, num_frames // row_n + 1, figsize=(40, 5))

print(ax.shape)
ax[0, 0].imshow(image_stack[0], cmap="gray", vmin=0, vmax=255)
ax[0, 0].set_title("Original image")
ax[0, 0].axis("off")

for n in range(1, len(rotated_image_stack) + 1):
    row = n % row_n
    col = n // row_n
    ax[row, col].imshow(
        rotated_image_stack[n - 1], cmap="gray", vmin=0, vmax=255
    )
    ax[row, col].set_title(n)

    # add information about the rotation angle
    angles = rotator.angles[n - 1]
    angle_range = f"{angles.min():.0f}-{angles.max():.0f}"
    ax[row, col].text(
        0.5,
        0.9,
        angle_range,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax[row, col].transAxes,
        color="white",
    )

#  last one, max projection
ax[row_n - 1, num_frames // row_n].imshow(
    rotated_image_stack.max(axis=0), cmap="gray"
)
#  plot also an x in the center of the image in red
ax[row_n - 1, num_frames // row_n].plot(
    rotated_image_stack.shape[2] / 2,
    rotated_image_stack.shape[1] / 2,
    "rx",
    markersize=10,
)
ax[row_n - 1, num_frames // row_n].set_title("Max projection")

for a in ax.ravel():
    a.axis("off")

plt.show()
