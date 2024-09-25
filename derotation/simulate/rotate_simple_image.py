import matplotlib.pyplot as plt
import numpy as np

from derotation.simulate.basic_rotator import Rotator

#  make a simple image, a white circle in a black background
image = np.zeros((100, 100))
y, x = np.ogrid[:100, :100]
mask = (x - 50) ** 2 + (y - 50) ** 2 < 20**2
image[mask] = 255


#  make a stack of 3 frames with the same image
image_stack = np.array([image, image, image])

#  make a list of angles, one per line per frame
num_angles = image_stack.shape[0] * image_stack.shape[1]
angles = np.arange(num_angles)

rotator = Rotator(angles, image_stack)
rotated_image_stack = rotator.rotate_by_line()

fig, ax = plt.subplots(1, 4, figsize=(20, 5))

ax[0].imshow(image, cmap="gray")
ax[0].set_title("Original image")
ax[0].axis("off")
for i, rotated_image in enumerate(rotated_image_stack):
    ax[i + 1].imshow(rotated_image, cmap="gray")
    ax[i + 1].set_title(f"Rotated image {i + 1}")
    ax[i + 1].axis("off")

plt.show()
