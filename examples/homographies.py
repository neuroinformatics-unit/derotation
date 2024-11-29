import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from skimage import transform

#  load example  image
img = ski.data.astronaut()

#  add a stripe of 100 pixels around as padding
n_pad = 0
img = np.pad(img, ((n_pad, n_pad), (n_pad, n_pad), (0, 0)), mode="constant")

#  image projected to a plane laying at 45 degrees from the x-y plane
angle_deg = 45

#  get the target dimensionality of the final projected
#  image based on the angle
angle_rad = np.radians(angle_deg)

#  define the homography matrix using the cosine of the angle
# in the x dimension
homography_matrix = np.array([[1, 0, 0], [0, np.cos(angle_rad), 0], [0, 0, 1]])

print(f"homography_matrix: {homography_matrix}")

inverse_homography_matrix = np.linalg.inv(homography_matrix)
print(f"inverse_homography_matrix: {inverse_homography_matrix}")

#  apply the homography
transformation_1 = transform.warp(img, inverse_homography_matrix)

#  make also inverse homography
transformation_2 = transform.warp(transformation_1, homography_matrix)

#  plot the images
fig, ax = plt.subplots(1, 3)
ax[0].imshow(img)
ax[0].set_title("Original image")
ax[1].imshow(transformation_1)
ax[1].set_title("Transformaion 1")
ax[2].imshow(transformation_2)
ax[2].set_title("Transformaion 2")

for a in ax:
    a.axis("off")

plt.show()
