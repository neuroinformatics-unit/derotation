"""Test"""

import matplotlib.pyplot as plt
import numpy as np

from derotation.derotate_by_line import derotate_an_image_array_line_by_line

path = "./images/dog.png"
img = plt.imread(path)
img = img[:, :, 0]

img_stack = np.array([[img, img, img]]).squeeze()

img_len = img.shape[0]
rotation_angles = np.linspace(0, 180, img_len * 3)

img_rotated = derotate_an_image_array_line_by_line(img_stack, rotation_angles)

fig, ax = plt.subplots(1, 4, figsize=(10, 5))

image_names = ["Original image"] + [f"Rotated image {i}" for i in range(1, 4)]
images_to_plot = [img] + [img_rotated[i] for i in range(3)]
for i, image_name in enumerate(image_names):
    ax[i].imshow(images_to_plot[i], cmap="gray")
    ax[i].set_title(image_name)
    ax[i].axis("off")

plt.show()
plt.close()

fig.savefig("./images/dog_rotated_by_line.png", bbox_inches="tight")
