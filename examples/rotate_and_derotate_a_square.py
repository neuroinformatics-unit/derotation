import matplotlib.pyplot as plt
import numpy as np

from derotation.derotate_by_line import derotate_an_image_array_line_by_line
from derotation.simulate.line_scanning_microscope import Rotator

#  make a simple image, a square in a black background
image = np.empty((100, 100))
#  it will have 5 gray levels to visualize better the rotation
gray_values = [i % 5 * 100 + 155 for i in range(100)]
for i in range(100):
    image[i] = gray_values[i]
#  make a black border
image[:20] = 0
image[-20:] = 0
image[:, :20] = 0
image[:, -20:] = 0


#  make a stack of 3 frames with the same image
image_stack = np.array([image, image, image])

#  make a list of angles, one per line per frame
num_angles = image_stack.shape[0] * image_stack.shape[1]
angles = np.arange(num_angles)

#  rotate the image stack
rotator = Rotator(angles, image_stack)
rotated_image_stack = rotator.rotate_by_line()

# now use derotation to revert to the original image
rotated_image_stack_derotated = derotate_an_image_array_line_by_line(
    rotated_image_stack, angles
)

#  plot the original image, the rotated images and the derotated images
fig, ax = plt.subplots(2, 4, figsize=(20, 5))

ax[0, 0].imshow(image, cmap="gray")
ax[0, 0].set_title("Original image")
ax[0, 0].axis("off")

for i, rotated_image in enumerate(rotated_image_stack):
    ax[0, i + 1].imshow(rotated_image, cmap="gray")
    ax[0, i + 1].set_title(f"Rotated image {i + 1}")
    ax[0, i + 1].axis("off")

    # add information about the rotation angle
    angles = rotator.angles[i]
    angle_range = f"angles: {angles.min():.0f}-{angles.max():.0f}"
    ax[0, i + 1].text(
        0.5,
        0.9,
        angle_range,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax[0, i + 1].transAxes,
        color="white",
    )

ax[1, 0].imshow(image, cmap="gray")
ax[1, 0].set_title("Original image")
ax[1, 0].axis("off")

for i, rotated_image in enumerate(rotated_image_stack_derotated):
    ax[1, i + 1].imshow(rotated_image, cmap="gray")
    ax[1, i + 1].set_title(f"Derotated image {i + 1}")
    ax[1, i + 1].axis("off")


plt.show()
