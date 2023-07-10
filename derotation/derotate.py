import copy
import pickle
from pathlib import Path

import numpy as np
import tifffile as tiff
from adjust_rotation_degrees import optimize_image_rotation_degrees
from find_centroid import (
    in_region,
    not_center_of_image,
    pipeline,
)
from matplotlib import pyplot as plt
from optimizers import find_best_k
from read_binary import read_rc2_bin
from scipy.io import loadmat
from scipy.ndimage import rotate
from scipy.signal import find_peaks

# Set aux and imaging locations and initialize dip image software
rot_deg = 360

path = Path("/Users/laura/data/230327_pollen")

path_tif = path / "imaging/runtest_00001.tif"
path_aux = path / "aux_stim/202303271657_21_005.bin"
path_config = "derotation/config.yml"
path_randperm = path / "stimlus_randperm.mat"

image = tiff.imread(path_tif)

pseudo_random = loadmat(path_randperm)
full_rotation_blocks_direction = pseudo_random["stimulus_random"][:, 2] > 0
dir = np.ones(5)
dir[full_rotation_blocks_direction[0:5]] = -1

data, dt, chan_names, config = read_rc2_bin(path_aux, path_config)
data_dict = {chan: data[:, i] for i, chan in enumerate(chan_names)}

frame_clock = data_dict["scanimage_frameclock"]
line_clock = data_dict["camera"]
full_rotation = data_dict["PI_rotCW"]
rotation_ticks = data_dict["Vistim_ttl"]
#  0.2 deg for 1 tick

# check if there is a missing frame
diffs = np.diff(frame_clock)
missing_frames = np.where(diffs > 0.1)[0]

# Calculate the threshold using a percentile of the total signal
best_k = find_best_k(frame_clock, image)
threshold = np.mean(frame_clock) + best_k * np.std(frame_clock)
print(f"Best threshold: {threshold}")
frames_start = np.where(np.diff(frame_clock) > threshold)[0]
frames_end = np.where(np.diff(frame_clock) < -threshold)[0]


#  find the peaks of the rot_tick2 signal
rot_tick2_peaks = find_peaks(
    rotation_ticks,
    height=4,
    distance=20,
)[0]


# sanity check for the number of rotation ticks
number_of_rotations = len(dir)
expected_tiks_per_rotation = rot_deg / dt
ratio = len(rot_tick2_peaks) / expected_tiks_per_rotation
if ratio > number_of_rotations:
    print(
        f"There are more rotation ticks than expected, {len(rot_tick2_peaks)}"
    )
elif ratio < number_of_rotations:
    print(
        f"There are less rotation ticks than expected, {len(rot_tick2_peaks)}"
    )


# identify the rotation ticks that correspond to
# clockwise and counter clockwise rotations
threshold = 0.5  # Threshold to consider "on" or rotation occurring
rotation_on = np.zeros_like(full_rotation)
rotation_on[full_rotation > threshold] = 1

#  delete intervals shorter than
rotation_signal_copy = copy.deepcopy(rotation_on)
latest_rotation_on_end = 0

i = 0
while i < len(dir):
    # find the first rotation_on == 1
    first_rotation_on = np.where(rotation_signal_copy == 1)[0][0]
    # now assign the value in dir to all the first set of ones
    len_first_group = np.where(rotation_signal_copy[first_rotation_on:] == 0)[
        0
    ][0]
    if len_first_group < 1000:
        #  skip this short rotation because it is a false one
        #  done one additional time to clean up the trace at the end
        rotation_signal_copy = rotation_signal_copy[
            first_rotation_on + len_first_group :
        ]
        latest_rotation_on_end = (
            latest_rotation_on_end + first_rotation_on + len_first_group
        )
        continue

    rotation_on[
        latest_rotation_on_end
        + first_rotation_on : latest_rotation_on_end
        + first_rotation_on
        + len_first_group
    ] = dir[i]
    latest_rotation_on_end = (
        latest_rotation_on_end + first_rotation_on + len_first_group
    )
    rotation_signal_copy = rotation_signal_copy[
        first_rotation_on + len_first_group :
    ]
    i += 1  # Increment the loop counter


#  calculate the rotation degrees for each frame
rotation_degrees = np.empty_like(frame_clock)
rotation_degrees[0] = 0
current_rotation: float = 0
tick_peaks_corrected = np.insert(rot_tick2_peaks, 0, 0, axis=0)
for i in range(0, len(tick_peaks_corrected)):
    time_interval = tick_peaks_corrected[i] - tick_peaks_corrected[i - 1]
    if time_interval > 2000 and i != 0:
        current_rotation = 0
    else:
        current_rotation += 0.2
    rotation_degrees[
        tick_peaks_corrected[i - 1] : tick_peaks_corrected[i]
    ] = current_rotation
signed_rotation_degrees = rotation_degrees * rotation_on
image_rotation_degree_per_frame = signed_rotation_degrees[frames_start]
image_rotation_degree_per_frame *= -1

try:
    with open("derotation/optimized_parameters.pkl", "rb") as f:
        optimized_parameters = pickle.load(f)
    with open("derotation/indexes.pkl", "rb") as f:
        indexes = pickle.load(f)
    with open("derotation/opt_result.pkl", "rb") as f:
        opt_result = pickle.load(f)
except FileNotFoundError:
    (
        opt_result,
        indexes,
        optimized_parameters,
    ) = optimize_image_rotation_degrees(image, image_rotation_degree_per_frame)
    with open("derotation/optimized_parameters.pkl", "wb") as f:
        pickle.dump(optimized_parameters, f)
    with open("derotation/indexes.pkl", "wb") as f:
        pickle.dump(indexes, f)
    with open("derotation/opt_result.pkl", "wb") as f:
        pickle.dump(opt_result, f)

new_image_rotation_degree_per_frame = copy.deepcopy(
    image_rotation_degree_per_frame
)
for i, block in enumerate(opt_result):
    new_image_rotation_degree_per_frame[indexes[i]] = block

#  make of indexes an unque flat array
indexes = np.unique(np.concatenate(indexes))


#  rotate the image to the correct position according to the frame_degrees
image = tiff.imread(path_tif)
rotated_image = np.empty_like(image)
rotated_image_corrected = np.empty_like(image)
centers = []
centers_rotated = []
centers_rotated_corrected = []
for i in range(len(image)):
    lower_threshold = -2700
    higher_threshold = -2600
    binary_threshold = 32
    sigma = 2.5

    defoulting_parameters = [
        lower_threshold,
        higher_threshold,
        binary_threshold,
        sigma,
    ]

    rotated_image[i] = rotate(
        image[i], image_rotation_degree_per_frame[i], reshape=False
    )
    rotated_image_corrected[i] = rotate(
        image[i], new_image_rotation_degree_per_frame[i], reshape=False
    )

    # params = optimized_parameters[i]
    # if i in indexes else defoulting_parameters

    centers.append(pipeline(image[i], defoulting_parameters))
    centers_rotated.append(pipeline(rotated_image[i], defoulting_parameters))
    centers_rotated_corrected.append(
        pipeline(rotated_image_corrected[i], defoulting_parameters)
    )

#  plot drift of centers
fig, ax = plt.subplots(3, 1)
for k, centroid in enumerate(centers):
    for c in centroid:
        if not_center_of_image(c) and in_region(c):
            ax[0].plot(k, c[1], marker="o", color="red")
            ax[0].plot(k, c[0], marker="o", color="blue")
            # ax[0].set_ylim(80, 180)
for k, centroid in enumerate(centers_rotated):
    for c in centroid:
        if not_center_of_image(c) and in_region(c):
            ax[1].plot(k, c[1], marker="o", color="red")
            ax[1].plot(k, c[0], marker="o", color="blue")
            # ax[0].set_ylim(80, 180)
for k, centroid in enumerate(centers_rotated_corrected):
    for c in centroid:
        if not_center_of_image(c) and in_region(c):
            ax[2].plot(k, c[1], marker="o", color="red")
            ax[2].plot(k, c[0], marker="o", color="blue")
            # ax[0].set_ylim(80, 180)

# Create a figure and axis for displaying the images
fig, ax = plt.subplots(1, 4)


ax[2].set_title("Rotation degrees per frame")

# Iterate through each image
for i, (image_rotated, image_original, image_corrected) in enumerate(
    zip(rotated_image, image, rotated_image_corrected)
):
    ax[0].imshow(image_original, cmap="gist_ncar")
    ax[1].imshow(image_rotated, cmap="gist_ncar")
    ax[2].imshow(image_corrected, cmap="gist_ncar")

    for c in centers[i]:
        if not_center_of_image(c) and in_region(c):
            # dim blob
            ax[0].plot(c[1], c[0], marker="*", color="red")
        if not not_center_of_image(c):
            # bright blob
            ax[0].plot(c[1], c[0], marker="*", color="white")
    for c in centers_rotated[i]:
        if not_center_of_image(c) and in_region(c):
            # dim blob
            ax[1].plot(c[1], c[0], marker="*", color="red")
        if not not_center_of_image(c):
            # bright blob
            ax[1].plot(c[1], c[0], marker="*", color="white")
    for c in centers_rotated_corrected[i]:
        if not_center_of_image(c) and in_region(c):
            # dim blob
            ax[2].plot(c[1], c[0], marker="*", color="red")
        if not not_center_of_image(c):
            # bright blob
            ax[2].plot(c[1], c[0], marker="*", color="white")

    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")

    # if len(centers[i]) > 0:
    #     ax[0].plot(
    #         centers[i][0][1], centers[i][0][0], marker="*", color="white"
    #     )
    # if len(centers[i]) > 1:
    #     ax[0].plot(centers[i][1][1],
    #               centers[i][1][0], marker="*", color="red")

    # ax[1].imshow(image_rotated, cmap="gist_ncar")
    # if len(centers_rotated[i]) > 0:
    #     ax[1].plot(
    #         centers_rotated[i][0][1],
    #         centers_rotated[i][0][0],
    #         marker="*",
    #         color="white",
    #     )
    # if len(centers_rotated[i]) > 1:
    #     ax[1].plot(
    #         centers_rotated[i][1][1],
    #         centers_rotated[i][1][0],
    #         marker="*",
    #         color="red",
    #     )

    #  add a vertical line on the plot on ax 2
    ax[3].axvline(frames_start[i], color="black", linestyle="--")
    ax[3].plot(signed_rotation_degrees, label="rotation degrees")
    ax[3].plot(
        frames_start,
        image_rotation_degree_per_frame,
        linestyle="none",
        marker="o",
        color="red",
    )

    plt.pause(0.001)
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()
    ax[3].clear()


ax[0].set_title("Original image")
ax[1].set_title("Rotated image")
ax[2].set_title("Corrected image")
ax[3].set_title("Rotation degrees per frame")

# axis off for the first two plots
ax[0].axis("off")
ax[1].axis("off")

# Close the figure
plt.close(fig)

# ==============================================================================
# PLOT
# ==============================================================================


fig, ax = plt.subplots(1, 1, sharex=True)
ax.boxplot(diffs)
ax.set_title("Threshold to identify frames start and end")
ax.set_ylabel("Difference between frames")

ax.axhline(threshold, 0, len(diffs), color="red", label="threshold")
ax.axhline(-threshold, 0, len(diffs), color="red", label="threshold")

fig, ax = plt.subplots(1, 1, sharex=True)
ax.plot(diffs, label="frame clock", color="black", alpha=0.5)

fig, ax = plt.subplots(4, 1, sharex=True)
ax[0].plot(
    frame_clock,
    label="frame clock",
    color="black",
    alpha=0.5,
    rasterized=True,
)
# plot dots for starting and ending points of the frame_clock signal
ax[0].plot(
    frames_start,
    frame_clock[frames_start],
    linestyle="none",
    marker="o",
    color="red",
    alpha=0.5,
    rasterized=True,
)
ax[0].plot(
    frames_end,
    frame_clock[frames_end],
    linestyle="none",
    marker="o",
    color="green",
    alpha=0.5,
    rasterized=True,
)
ax[1].plot(
    line_clock,
    label="line clock",
    color="red",
    alpha=0.5,
    rasterized=True,
)
ax[2].plot(
    full_rotation,
    label="rot tick",
    color="blue",
    alpha=0.5,
    rasterized=True,
)
ax[2].plot(
    rotation_on,
    label="rotation with direction, 1 = CW, -1 = CCW",
    color="green",
    alpha=0.5,
    rasterized=True,
)
ax[3].plot(
    rotation_ticks,
    label="rot tick 2",
    color="green",
    alpha=0.5,
    marker="o",
    rasterized=True,
)
ax[3].plot(
    rot_tick2_peaks,
    # np.ones(len(rot_tick2_peaks)) * 5.2,
    rotation_ticks[rot_tick2_peaks],
    linestyle="none",
    marker="*",
    color="red",
    alpha=0.5,
    rasterized=True,
)


# set the initial x axis limits
for axis in ax:
    # axis.set_xlim(1610000, 1800000)
    # axis.set_xlim(1680000, 1710000)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

#  set subplots titles
ax[0].set_title("Frame clock (black) and starting/ending points (red/green)")
ax[1].set_title("Line clock")
ax[2].set_title("Full rotation info")
ax[3].set_title("Rotation ticks, 0.2 deg for 1 tick (green), peaks (red)")

#  plot title
fig.suptitle("Frame clock and rotation ticks")

plt.show()
