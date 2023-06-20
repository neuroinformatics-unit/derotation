from pathlib import Path

import numpy as np
import tifffile as tiff
from matplotlib import pyplot as plt
from read_binary import read_rc2_bin
from scipy.io import loadmat
from scipy.signal import find_peaks
from optimizers import find_best_k

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
frames_start = np.where(np.diff(frame_clock) > threshold)[0]
frames_end = np.where(np.diff(frame_clock) < -threshold)[0]

fig, ax = plt.subplots(1, 1, sharex=True)
ax.boxplot(diffs)
ax.set_title("Boxplot of the frame clock signal")
ax.set_ylabel("Difference between frames")

ax.axhline(threshold, 0, len(diffs), color="red", label="threshold")
ax.axhline(-threshold, 0, len(diffs), color="red", label="threshold")


# fig, ax = plt.subplots(1, 1, sharex=True)
# ax.plot(diffs, label="frame clock", color="black", alpha=0.5)


#  find the peaks of the rot_tick2 signal
rot_tick2_peaks = find_peaks(rotation_ticks, height=4)[0]
#  exclude external ticks

#  identify rotation blocks
rotation_blocks = np.where(full_rotation > 4)

#  assign to each block a number in order to identify the direction


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
    axis.set_xlim(1680000, 1710000)
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

print("test")
