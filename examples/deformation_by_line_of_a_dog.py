from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from derotation.derotate_by_line import rotate_an_image_array_line_by_line

path = Path(__file__).parent.parent / "images/dog.png"
img = plt.imread(path)
img = img[:, :, 0]

img_stack = np.array([[img, img, img]]).squeeze()

img_len = img.shape[0]
rotation_angles = np.linspace(0, 180, img_len * 3)

img_rotated = rotate_an_image_array_line_by_line(img_stack, rotation_angles)

fig, ax = plt.subplots(1, 4, figsize=(10, 5))

ax[0].imshow(img, cmap="gray")
ax[0].set_title("Original image")
ax[1].imshow(img_rotated[0], cmap="gray")
ax[1].set_title("Rotated image 1")
ax[2].imshow(img_rotated[1], cmap="gray")
ax[2].set_title("Rotated image 2")
ax[3].imshow(img_rotated[2], cmap="gray")
ax[3].set_title("Rotated image 3")

ax[0].axis("off")
ax[1].axis("off")
plt.show()

fig.savefig("dog_rotated_by_line.png")
