import copy
from pathlib import Path

import numpy as np
import tifffile as tiff
from matplotlib import pyplot as plt
from optimizers import find_best_k
from read_binary import read_rc2_bin
from scipy.io import loadmat
from scipy.signal import find_peaks

# Set aux and imaging locations and initialize dip image software
rot_deg = 360

path = Path("/Users/lauraporta/local_data/230327_pollen")

path_tif = path / "imaging/runtest_00001.tif"
path_aux = path / "aux_stim/202303271657_21_005.bin"
path_config = "derotation/config.yml"
path_randperm = path / "stimlus_randperm.mat"

image = tiff.imread(path_tif)

pseudo_random = loadmat(path_randperm)
full_rotation_blocks_direction = pseudo_random["stimulus_random"][:, 2] > 0
dir = np.ones(4)
dir[full_rotation_blocks_direction[0:4]] = -1

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

# fig, ax = plt.subplots(1, 1, sharex=True)
# ax.boxplot(diffs)
# ax.set_title("Threshold to identify frames start and end")
# ax.set_ylabel("Difference between frames")

# ax.axhline(threshold, 0, len(diffs), color="red", label="threshold")
# ax.axhline(-threshold, 0, len(diffs), color="red", label="threshold")

# fig, ax = plt.subplots(1, 1, sharex=True)
# ax.plot(diffs, label="frame clock", color="black", alpha=0.5)

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


# identify the rotation ticks that correspond to clockwise and counter clockwise rotations
threshold = 0.5  # Threshold to consider "on" or rotation occurring
rotation_on = np.zeros_like(full_rotation)
rotation_on[full_rotation > threshold] = 1

rotation_signal_copy = copy.deepcopy(rotation_on)
latest_rotation_on_end = 0
for i in range(len(dir)):
    # find the first rotation_on == 1
    first_rotation_on = np.where(rotation_signal_copy == 1)[0][0]
    # now assign the value in dir to all the first set of ones
    len_first_group = np.where(rotation_signal_copy[first_rotation_on:] == 0)[
        0
    ][0]
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


# every tick is an increment of 0.2 deg
rotation_degrees = np.empty_like(
    frame_clock
)  # Initialize the rotation degree array with the initial value
rotation_degrees[0] = 0
current_rotation = 0
tick_peaks_corrected = np.insert(rot_tick2_peaks, 0, 0, axis=0)

for i in range(0, len(tick_peaks_corrected)):
    time_interval = tick_peaks_corrected[i] - tick_peaks_corrected[i - 1]
    if time_interval > 10000 and i != 0:
        current_rotation = 0
    current_rotation += dt
    rotation_degrees[tick_peaks_corrected[i]] = current_rotation

only_rotations = rotation_degrees[rotation_degrees != 0]
assert len(tick_peaks_corrected) == len(
    only_rotations
), f"{len(tick_peaks_corrected)} != {len(only_rotations)}"

signed_rotation_degrees = rotation_degrees * rotation_on
# signed_only_rotations = signed_rotation_degrees[signed_rotation_degrees != 0]
# assert len(tick_peaks_corrected) == len(signed_only_rotations), f"{len(tick_peaks_corrected)} != {len(signed_only_rotations)}"
#  if this fails it means that there are some rotation ticks outside of the rotation blocks

signed_rotation_degrees[np.isclose(signed_rotation_degrees, 0)] = np.nan

xvals = np.arange(0, len(rotation_ticks))
yinterp = np.interp(xvals, rotation_ticks, signed_rotation_degrees)


#  we can consider the frames as indexes of the rotation_degrees array
rotation_degrees_frames = np.empty_like(frame_clock)
rotation_degrees_frames[frames_start] = rotation_degrees[frames_start]


fig, ax = plt.subplots(1, 1, sharex=True)
ax.plot(
    rotation_degrees_frames, label="rotation degrees", color="black", alpha=0.5
)
ax.plot(
    signed_rotation_degrees, label="rotation degrees", color="red", alpha=0.5
)

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
