# %%
from pathlib import Path

import allensdk.brain_observatory.dff as dff_module
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from allensdk.brain_observatory.r_neuropil import NeuropilSubtract
from scipy.io import loadmat

saving_path = Path(
    "/Users/lauraporta/Source/github/neuroinformatics-unit/derotation/examples/figures/pdf/"
)


def neuropil_subtraction(f, f_neu):
    #  use default parameters for all methods
    neuropil_subtraction = NeuropilSubtract()
    neuropil_subtraction.set_F(f, f_neu)
    neuropil_subtraction.fit()

    r = neuropil_subtraction.r

    f_corr = f - r * f_neu

    # kernel values to be changed for 3-photon data
    # median_kernel_long = 1213, median_kernel_short = 23

    dff = 100 * dff_module.compute_dff_windowed_median(f_corr)

    return dff, r


F_path = Path(
    "/Users/lauraporta/local_data/rotation/230802_CAA_1120182/full/derotated/CE/suite2p/plane0/F.npy"
)
f = np.load(F_path)
print(f.shape)

Fneu_path = Path(
    "/Users/lauraporta/local_data/rotation/230802_CAA_1120182/full/derotated/CE/suite2p/plane0/Fneu.npy"
)
fneu = np.load(Fneu_path)
print(fneu.shape)

dff, r = neuropil_subtraction(
    f=f,
    f_neu=fneu,
)

dff = pd.DataFrame(dff.T)
print(dff.shape)
print(dff.head())


path_randperm = Path(
    "/Users/lauraporta/local_data/rotation/stimlus_randperm.mat"
)
pseudo_random = loadmat(path_randperm)
rotation_speed = pseudo_random["stimulus_random"][:, 0]

full_rotation_blocks_direction = pseudo_random["stimulus_random"][:, 2] > 0
direction = np.where(
    full_rotation_blocks_direction, -1, 1
)  # 1 is counterclockwise, -1 is clockwise


rotated_frames_path = Path(
    "/Users/lauraporta/local_data/rotation/230802_CAA_1120182/full/derotated/CE/derotated_image_stack_CE.csv"
)
rotated_frames = pd.read_csv(rotated_frames_path)
print(rotated_frames.head())
print(rotated_frames.shape)

full_dataframe = pd.concat([dff, rotated_frames], axis=1)

# %%

subset = full_dataframe[
    (full_dataframe["speed"] == 100) & (full_dataframe["direction"] == -1)
]

rois_selection = [4, 8, 14, 20, 23]

merged_mean = pd.DataFrame()
for roi in rois_selection:
    mean_response = subset.loc[:, [roi, "rotation_count"]]
    mean_response["counter"] = np.zeros(len(mean_response)) - 1
    latest_rotation = -1
    for idx in mean_response.index:
        if mean_response.loc[idx, "rotation_count"] > latest_rotation:
            counter = 0
            latest_rotation = mean_response.loc[idx, "rotation_count"]
        elif mean_response.loc[idx, "rotation_count"] == latest_rotation:
            counter += 1
        mean_response.loc[idx, "counter"] = counter

    mean = mean_response.groupby("counter").mean()

    # plt.plot(mean.loc[:, [roi]])

    merged_mean[f"roi_{roi}"] = mean.loc[:, [roi]]

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
ax = sns.lineplot(merged_mean)
ax.set(xlabel="Frames during rotation (100 deg/s)", ylabel="ΔF/F")


print("debug")

# %%

# find where do rotations start
rotation_on = np.diff(full_dataframe["rotation_count"])


def find_zero_chunks(arr):
    zero_chunks = []
    start = None

    for i in range(len(arr)):
        if arr[i] == 0 and start is None:
            start = i
        elif arr[i] != 0 and start is not None:
            zero_chunks.append((start, i - 1))
            start = None

    # Check if the array ends with a chunk of zeros
    if start is not None:
        zero_chunks.append((start, len(arr) - 1))

    return zero_chunks


starts_ends = find_zero_chunks(rotation_on)

frames_before_rotation = 10
# frames_after_rotation = 10

total_len = 70

full_dataframe["rotation_frames"] = np.zeros(len(full_dataframe))
for i, (start, end) in enumerate(starts_ends):
    frame_array = np.arange(total_len)
    column_index_of_rotation_frames = full_dataframe.columns.get_loc(
        "rotation_frames"
    )
    full_dataframe.iloc[
        start
        - frames_before_rotation : total_len
        + start
        - frames_before_rotation,
        column_index_of_rotation_frames,
    ] = frame_array

    #  extend this value of speed and direction to all this range
    this_speed = full_dataframe.loc[start, "speed"]
    this_direction = full_dataframe.loc[start, "direction"]

    full_dataframe.iloc[
        start
        - frames_before_rotation : total_len
        + start
        - frames_before_rotation,
        full_dataframe.columns.get_loc("speed"),
    ] = this_speed
    full_dataframe.iloc[
        start
        - frames_before_rotation : total_len
        + start
        - frames_before_rotation,
        full_dataframe.columns.get_loc("direction"),
    ] = this_direction


#  directtion, change -1 to CCW and 1 to CW
full_dataframe["direction"] = np.where(
    full_dataframe["direction"] == -1, "CCW", "CW"
)


# %%
#  Single traces for every ROI
selected_range = (400, 2000)

for roi in range(11):
    roi_selected = full_dataframe.loc[
        :, [roi, "rotation_count", "speed", "direction"]
    ]

    fig, ax = plt.subplots(figsize=(27, 5))
    ax.plot(roi_selected.loc[selected_range[0] : selected_range[1], roi])
    ax.set(xlabel="Frames", ylabel="ΔF/F")

    rotation_on = (
        np.diff(
            roi_selected.loc[
                selected_range[0] : selected_range[1], "rotation_count"
            ]
        )
        == 0
    )

    # add label at the beginning of every block of rotations
    #  if the previous was true, do not write the label
    for i, rotation in enumerate(rotation_on):
        if rotation and not rotation_on[i - 1]:
            ax.text(
                i + selected_range[0] + 3,
                -500,
                f"{int(roi_selected.loc[i + 5 + selected_range[0], 'speed'])}º/s\n{roi_selected.loc[i + 5 + selected_range[0], 'direction']}",
                fontsize=10,
            )

    #  add gray squares when the rotation is happening using the starst_ends
    for start, end in starts_ends:
        if start > selected_range[0] and end < selected_range[1]:
            ax.axvspan(start, end, color="gray", alpha=0.2)

    fps = 6.74
    # change xticks to seconds
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels((xticks / fps).astype(int))
    #  change x label
    ax.set(xlabel="Seconds", ylabel="ΔF/F")

    ax.set_xlim(selected_range)
    ax.set_ylim(-300, 300)

    # leave some gap between the axis and the plot
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig(saving_path / f"dff_example_{roi}.pdf")
    plt.close()


# %%
rois_selection = list(range(11))

#  do it with seaborn
for roi in rois_selection:
    fig, ax = plt.subplots(2, 2, figsize=(27, 10))
    for i, speed in enumerate([50, 100, 150, 200]):
        sns.lineplot(
            x="rotation_frames",
            y=roi,
            data=full_dataframe[(full_dataframe["speed"] == speed)],
            hue="direction",
            ax=ax[i // 2, i % 2],
        )
        ax[i // 2, i % 2].set(xlabel="Frames", ylabel="ΔF/F")
        ax[i // 2, i % 2].set_title(f"Speed: {speed}º/s")
        ax[i // 2, i % 2].legend(title="Direction")

        #  remove top and right spines
        ax[i // 2, i % 2].spines["top"].set_visible(False)
        ax[i // 2, i % 2].spines["right"].set_visible(False)

        # add vertical lines to show the start of the rotation
        #  start is always at 11, end at total len - 10
        ax[i // 2, i % 2].axvline(
            x=frames_before_rotation, color="gray", linestyle="--"
        )
        this_x_axis_len = ax[i // 2, i % 2].get_xlim()[1]
        ax[i // 2, i % 2].axvline(
            x=this_x_axis_len - frames_before_rotation,
            color="gray",
            linestyle="--",
        )
        #  to seconds
        fps = 6.74
        xticks = ax[i // 2, i % 2].get_xticks()
        ax[i // 2, i % 2].set_xticks(xticks)
        ax[i // 2, i % 2].set_xticklabels(np.round(xticks / fps, 1))
        #  change x label
        ax[i // 2, i % 2].set(xlabel="Seconds", ylabel="ΔF/F")

        #  unique y scale for all
        # ax[i // 2, i % 2].set_ylim(-100, 100)

    plt.savefig(saving_path / f"roi_{roi}_speed_direction.pdf")
    plt.close()

# %%
#  now another similar plot. two subplots. On the left CW, right CCW
#  hue is speed.
rois_selection = list(range(11))


custom_palette = sns.color_palette("dark:#5A9_r", 4)

for roi in rois_selection:
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    for i, direction in enumerate(["CW", "CCW"]):
        sns.lineplot(
            x="rotation_frames",
            y=roi,
            data=full_dataframe[(full_dataframe["direction"] == direction)],
            hue="speed",
            palette=custom_palette,
            ax=ax[i],
        )
        ax[i].set_title(f"Direction: {direction}")
        ax[i].legend(title="Speed")

        #  remove top and right spines
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)

        # add vertical lines to show the start of the rotation
        #  start is always at 11, end at total len - 10
        ax[i].axvline(x=frames_before_rotation, color="gray", linestyle="--")

        #  change x axis to seconds
        fps = 6.74
        xticks = ax[i].get_xticks()
        ax[i].set_xticks(xticks)
        ax[i].set_xticklabels(np.round(xticks / fps, 1))
        #  change x label
        ax[i].set(xlabel="Seconds", ylabel="ΔF/F")

    plt.savefig(saving_path / f"roi_{roi}_direction_speed.pdf")
    plt.close()


# %%
# Same plot as above but only for clockwise rotations
#  columns: different speeds
#  rows: different rois [0, 8, 9]

rois_selection = [0, 8, 9]
custom_palette = sns.color_palette("dark:#5A9_r", 4)

for roi in rois_selection:
    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    for i, speed in enumerate([50, 100, 150, 200]):
        sns.lineplot(
            x="rotation_frames",
            y=roi,
            data=full_dataframe[
                (full_dataframe["direction"] == "CW")
                & (full_dataframe["speed"] == speed)
            ],
            ax=ax[i],
        )
        ax[i].set_title(f"Speed: {speed}º/s")
        ax[i].legend(title="Direction")

        #  remove top and right spines
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)

        # add vertical lines to show the start of the rotation
        #  start is always at 11, end at total len - 10
        ax[i].axvline(x=frames_before_rotation, color="gray", linestyle="--")

        # gray box

        #  change x axis to seconds
        fps = 6.74
        xticks = ax[i].get_xticks()
        ax[i].set_xticks(xticks)
        ax[i].set_xticklabels(np.round(xticks / fps, 1))
        #  change x label
        ax[i].set(xlabel="Seconds", ylabel="ΔF/F")

        #  ylim: -70 : 320
        ax[i].set_ylim(-70, 320)

    plt.savefig(saving_path / f"roi_{roi}_CW_speed.pdf")
    plt.close()

#  for the same selection, take the peak response and the std at the peak and plot them together in
#  a scatter plot (x: speed, y: peak response, color: roi)

#  get the peak response and the std at the peak
peak_response = pd.DataFrame()
for roi in rois_selection:
    for speed in [50, 100, 150, 200]:
        subset = full_dataframe[
            (full_dataframe["direction"] == "CW")
            & (full_dataframe["speed"] == speed)
        ]
        all_peaks = subset.loc[:, roi].max()
        peak_response = peak_response.append(
            {
                "roi": roi,
                "speed": speed,
                "peak_response": np.max(all_peaks),
                "std": subset.loc[:, roi].std(),
            },
            ignore_index=True,
        )


fig, ax = plt.subplots(figsize=(20, 10))
sns.scatterplot(
    x="speed",
    y="peak_response",
    data=peak_response,
    hue="roi",
    ax=ax,
)

ax.set(xlabel="Speed (º/s)", ylabel="Peak response (ΔF/F)")
ax.legend(title="ROI")
#  connect the dots
for roi in rois_selection:
    roi_data = peak_response[peak_response["roi"] == roi]
    for i in range(0, 4):
        ax.plot(
            roi_data.loc[roi_data.index[i - 1 : i + 1], "speed"],
            roi_data.loc[roi_data.index[i - 1 : i + 1], "peak_response"],
            color="gray",
            linestyle="--",
        )

        # add the std at the peak as error bars
        ax.errorbar(
            roi_data.loc[roi_data.index[i], "speed"],
            roi_data.loc[roi_data.index[i], "peak_response"],
            yerr=roi_data.loc[roi_data.index[i], "std"],
            fmt="o",
            color="gray",
        )

#  remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


plt.savefig(saving_path / "peak_response_speed.pdf")


# %%

#  now similar plot but according to rotation angle
#  for every angle, get the mean of the roi and plot it

angles_to_pick = [0, 45, 90, 135, 180, 225, 270, 315]


tollerace_deg = 1

dataframe_copy_with_rounded_angles = full_dataframe.copy()
dataframe_copy_with_rounded_angles["rotation_angle"] = np.round(
    full_dataframe["rotation_angle"]
)
#  now np.abs
dataframe_copy_with_rounded_angles.loc[:, "rotation_angle"] = np.abs(
    dataframe_copy_with_rounded_angles["rotation_angle"]
)

for roi in rois_selection:
    fig, ax = plt.subplots(figsize=(20, 10))
    mean_response_per_angle = (
        dataframe_copy_with_rounded_angles.loc[:, [roi, "rotation_angle"]]
        .groupby("rotation_angle")
        .mean()
    )
    std = (
        dataframe_copy_with_rounded_angles.loc[:, [roi, "rotation_angle"]]
        .groupby("rotation_angle")
        .std()
    )

    sns.lineplot(
        x="rotation_angle", y=roi, data=mean_response_per_angle, ax=ax
    )

    ax.fill_between(
        mean_response_per_angle.index,
        mean_response_per_angle[roi] - std[roi],
        mean_response_per_angle[roi] + std[roi],
        alpha=0.2,
    )
    ax.set(xlabel="Rotation angle (º)", ylabel="ΔF/F")
    ax.set_title(f"ROI {roi}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig(saving_path / f"roi_{roi}_rotation_angle.pdf")

# %%
