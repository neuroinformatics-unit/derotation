from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from skimage.registration import phase_cross_correlation

tif_path = Path(
    "/Users/laura/data/derotation/raw/230802_CAA_1120182/incremental/derotated/derotated_masked_full_incremental.tif"
)
tif = tifffile.imread(tif_path)

csv_path = Path(
    "/Users/laura/data/derotation/raw/230802_CAA_1120182/incremental/derotated/derotated_masked_full_incremental.csv"
)
df = pd.read_csv(csv_path, index_col=0, header=0)
# aproximate angles with 0 decimals
df["rotation_angle"] = df["rotation_angle"] - 0.01
df["rotation_angle"] = df["rotation_angle"].round(2)

# for rotation degrees 10, 20, 30... make mean images
mean_images = []

for i in range(0, 360, 10):
    df_idx_angle = df[df["rotation_angle"] == i]
    images = tif[df_idx_angle.index]

    mean_image = np.mean(images, axis=0)

    mean_images.append(mean_image)

#  drop the first
mean_images = mean_images[1:]

image = np.mean(tif[:100], axis=0)

peaks = []
for i, offset_image in enumerate(mean_images):
    shift, error, diffphase = phase_cross_correlation(image, offset_image)

    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(1, 3, 3)

    ax1.imshow(image, cmap="gray")
    ax1.set_axis_off()
    ax1.set_title("Reference image")

    ax2.imshow(offset_image.real, cmap="gray")
    ax2.set_axis_off()
    ax2.set_title("Offset image")

    # Show the output of a cross-correlation to show what the algorithm is
    # doing behind the scenes
    image_product = np.fft.fft2(image) * np.fft.fft2(offset_image).conj()
    cc_image = np.fft.fftshift(np.fft.ifft2(image_product))

    peaks.append(np.unravel_index(np.argmax(cc_image), cc_image.shape))

    ax3.imshow(cc_image.real)
    ax3.set_axis_off()
    ax3.set_title("Cross-correlation")

    # save plot
    fig.savefig(
        f"/Users/laura/data/derotation/raw/230802_CAA_1120182/incremental/phase_cross_corr/pixel_precision_{i}.png"
    )


df = pd.DataFrame(peaks, columns=["x", "y"])
df["rotation_angle"] = np.arange(10, 360, 10)
df.to_csv(
    "/Users/laura/data/derotation/raw/230802_CAA_1120182/incremental/phase_cross_corr/pixel_precision.csv"
)

#  plot the shift values
fig, ax = plt.subplots()
ax.plot(df["rotation_angle"], df["x"], label="x")
ax.plot(df["rotation_angle"], df["y"], label="y")
ax.set_xlabel("rotation angle")
ax.set_ylabel("shift value")
ax.legend()

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

fig.savefig(
    "/Users/laura/data/derotation/raw/230802_CAA_1120182/incremental/phase_cross_corr/pixel_precision_shift_values.png"
)

image_center = np.array(image.shape) / 2
shifts = df[["x", "y"]]
shifts = shifts - image_center
shifts = shifts.astype(int)

# now use the shift values to correct the position of the images
registered_images = []
for i, image in enumerate(mean_images):
    # shift = df.iloc[i][["x", "y"]]
    # shift = shift - image_center
    # shift = shift.astype(int)
    registered_image = np.roll(image, shift=shifts.iloc[i], axis=(0, 1))
    registered_images.append(registered_image)

registered_images = np.array(registered_images)
tifffile.imwrite(
    "/Users/laura/data/derotation/raw/230802_CAA_1120182/incremental/derotated/derotated_masked_full_incremental_registered_small.tif",
    registered_images,
)
