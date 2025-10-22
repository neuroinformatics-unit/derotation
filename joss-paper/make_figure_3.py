# To reproduce the figure, run the following commands from the current
# directory
#
# conda create -n derotation-env python=3.12 -y
# conda activate derotation-env
# pip install photon-mosaic==0.2.1
# python make_figure_3.py

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import affine_transform
from sklearn.mixture import GaussianMixture

from derotation.analysis.full_derotation_pipeline import FullPipeline
from derotation.config.load_config import load_config, update_config_paths
from derotation.sample_data import fetch_data

# =====================================================
# LOAD DATA AND DEROTATE BY LINE AND BY FRAME
# =====================================================

color_line_based = "hotpink"
color_frame_based = "darkslategray"
rotation_on_color = "black"

current_module_path = Path.cwd()

# downloading the data might take a while the first time
config = load_config()
config = update_config_paths(
    config=config,
    tif_path=str(fetch_data("figure3/rotated_stack.tif")),
    aux_path=str(fetch_data("figure3/analog_signals.bin")),
    stim_randperm_path=str(
        fetch_data("figure3/stimlus_random_permutations.mat")
    ),
    output_folder=str(current_module_path),
)

# create and run the derotation pipeline
pipeline = FullPipeline(config)

pipeline()  # derotation by line
# it will take a while as it will run the center optimization

# get relevant variables from the pipeline
image_stack = pipeline.image_stack
angles_per_frame = pipeline.rot_deg_frame
angles_per_line = pipeline.rot_deg_line
rotation_center = pipeline.center_of_rotation
blank_pixels_value = pipeline.offset
derotated_stack = pipeline.masked_image_volume

angles_per_frame = angles_per_frame[: image_stack.shape[0]]
framerate = config["frame_rate"]

#  rotate by frame around the specified rotation center
center = np.array(rotation_center)
derotated_stack_frame = np.empty_like(image_stack)

for i, angle in enumerate(angles_per_frame):
    angle_rad = -np.deg2rad(angle)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    offset = center - R @ center

    # apply the affine transformation to derotate by frame
    derotated_stack_frame[i] = affine_transform(
        image_stack[i],
        R,
        offset=offset,
        output_shape=derotated_stack[i].shape,
        order=0,
        mode="constant",
        cval=blank_pixels_value,
    )

# =====================================================
# RUN SUITE2P VIA PHOTON-MOSAIC ON BOTH DEROTATED STACKS
# =====================================================

#  save the two image stacks here as tif files
os.makedirs("./output/dataset0/", exist_ok=True)
os.makedirs("./output/dataset1/", exist_ok=True)
tifffile.imwrite("./output/dataset0/stack_by_line.tif", derotated_stack)
tifffile.imwrite("./output/dataset1/stack_by_frame.tif", derotated_stack_frame)


#  Now let's run suite2p via photon-mosaic to do image registration and
#  signal extraction
cmd = "photon-mosaic --raw_data_base ./output/ --processed_data_base ./"
#  dry run to see what will be done
os.system(cmd + " --dry-run")
#  in case you want to change suite2p configs do it before running the command
os.system(cmd)

# =====================================================
# LOAD RESULTS AND PLOT FIGURE 3
# =====================================================

derotation_csv = "derotated.csv"
derotation_df = pd.read_csv(derotation_csv)
rotation_on = np.where(
    np.abs(derotation_df["rotation_angle"].values) > 0.1, 1, np.nan
)

derotated_tiff_path = "derotated.tif"
derotated_by_line = tifffile.imread(derotated_tiff_path)

derotated_by_frame_path = "./output/dataset1/stack_by_frame.tif"
derotated_by_frame = tifffile.imread(derotated_by_frame_path)

path_to_bin_file = (
    "./derivatives/sub-0_dataset0/ses-0/funcimg/suite2p/plane0/data.bin"
)
registered = np.memmap(
    path_to_bin_file, shape=derotated_by_line.shape, dtype="int16"
)

path_to_bin_file_frame = (
    "./derivatives/sub-1_dataset1/ses-0/funcimg/suite2p/plane0/data.bin"
)
registered_frame = np.memmap(
    path_to_bin_file_frame, shape=derotated_by_frame.shape, dtype="int16"
)

# normalize registered images for visualization
percentile_start = 97
percentile_end = 100

registered = np.clip(
    registered,
    np.percentile(registered, percentile_start),
    np.percentile(registered, percentile_end),
)
registered = (registered - registered.min()) / (
    registered.max() - registered.min()
)

# suite2p stat paths for the two processed datasets
stat_path_line = (
    "./derivatives/sub-0_dataset0/ses-0/funcimg/suite2p/plane0/stat.npy"
)
stat_path_frame = (
    "./derivatives/sub-1_dataset1/ses-0/funcimg/suite2p/plane0/stat.npy"
)

# load fluorescence traces F and Fneu
path_F_stack_by_line = (
    "./derivatives/sub-0_dataset0/ses-0/funcimg/suite2p/plane0/F.npy"
)
path_F_stack_by_frame = (
    "./derivatives/sub-1_dataset1/ses-0/funcimg/suite2p/plane0/F.npy"
)
path_F_neu_stack_by_line = (
    "./derivatives/sub-0_dataset0/ses-0/funcimg/suite2p/plane0/Fneu.npy"
)
path_F_neu_stack_by_frame = (
    "./derivatives/sub-1_dataset1/ses-0/funcimg/suite2p/plane0/Fneu.npy"
)

F_by_line = np.load(path_F_stack_by_line)
F_by_frame = np.load(path_F_stack_by_frame)
F_neu_by_line = np.load(path_F_neu_stack_by_line)
F_neu_by_frame = np.load(path_F_neu_stack_by_frame)

# neuropil subtracted fluorescence
sub_line = F_by_line - F_neu_by_line
sub_frame = F_by_frame - F_neu_by_frame


def df_over_f(f):
    #  in this case we just take the mean of the first 100 frames
    gmm = GaussianMixture(n_components=7, random_state=0).fit(f.reshape(-1, 1))
    first_component = gmm.means_[0]
    return (f - first_component) / first_component


# now let's plot figure 3
fig = plt.figure(figsize=(13, 5))

#  axis definitions
red_circles_plot = plt.subplot2grid((1, 10), (0, 0), rowspan=1, colspan=4)
line_plot = plt.subplot2grid((2, 10), (0, 4), rowspan=1, colspan=6)
mean_plot_CW = plt.subplot2grid((2, 10), (1, 4), rowspan=1, colspan=3)
mean_plot_CCW = plt.subplot2grid((2, 10), (1, 7), rowspan=1, colspan=3)

# here the median position of the ROI we want to plot]
choosen_roi = [27, 155]

# plot the mean image with the chosen ROI highlighted
red_circles_plot.imshow(np.mean(registered, axis=0), cmap="turbo")
red_circles_plot.axis("off")
red_circles_plot.scatter(
    choosen_roi[0],
    choosen_roi[1],
    facecolor="none",
    edgecolor=color_line_based,
    s=300,
    marker="o",
)


#  find the closest ROI to the median points in both datasets
def find_closest_roi(stat_path, target_xy):
    stats = np.load(stat_path, allow_pickle=True)
    centers = np.array(
        [[s["med"][1], s["med"][0]] for s in stats], dtype=float
    )
    idx = int(
        np.argmin(
            np.linalg.norm(
                centers - np.array(target_xy, dtype=float)[None, :], axis=1
            )
        )
    )
    return idx, centers[idx]


cell_ids_by_line, centroid_line = find_closest_roi(stat_path_line, choosen_roi)
cell_ids_by_frame, centroid_frame = find_closest_roi(
    stat_path_frame, choosen_roi
)

print(
    f"Matched ROI for line-based dataset: index={cell_ids_by_line}, "
    f"centroid={centroid_line}"
)
print(
    f"Matched ROI for frame-based dataset: index={cell_ids_by_frame}, "
    f"centroid={centroid_frame}"
)

number_of_frames_to_plot = 1100

df_over_f_line = df_over_f(sub_line[cell_ids_by_line])
df_over_f_frame = df_over_f(sub_frame[cell_ids_by_frame])
time_range = [4500, 4500 + number_of_frames_to_plot]

line_plot.plot(
    df_over_f_frame[time_range[0] : time_range[1]],
    color=color_frame_based,
    linewidth=1.5,
)
line_plot.plot(
    df_over_f_line[time_range[0] : time_range[1]],
    color=color_line_based,
    linewidth=1.5,
)
line_plot.set_ylabel("ΔF/F$_0$")
this_rotation_on = rotation_on[time_range[0] : time_range[1]]
line_plot.plot(
    this_rotation_on * 2,
    color=rotation_on_color,
    linewidth=4,
    alpha=1,
    solid_capstyle="butt",
)

this_rotation_on = rotation_on[time_range[0] : time_range[1]]

line_plot.spines["top"].set_visible(False)
line_plot.spines["right"].set_visible(False)

line_plot.set_xlabel("Frames")

#  now plot mean traces for CW and CCW rotations separately
speeds = derotation_df["speed"].values
unique_speeds = np.unique(derotation_df["speed"].dropna().values.astype(float))


for axis, direction in zip([mean_plot_CCW, mean_plot_CW], [-1.0, 1.0]):
    speed = 100
    rotation_times = np.where(
        (derotation_df["speed"].values == speed)
        & (derotation_df["direction"] == direction)
    )[0]
    rotation_segments = np.split(
        rotation_times, np.where(np.diff(rotation_times) != 1)[0] + 1
    )

    #  for each segment, extract the fluorescence traces and average them
    df_over_f_line = df_over_f(sub_line[cell_ids_by_line])
    df_over_f_frame = df_over_f(sub_frame[cell_ids_by_frame])

    #  for each segment, extract the fluorescence traces aligned to rotation
    #  onset use a fixed pre/post window so all traces have the same length
    #  and can be averaged
    line_traces = []
    frame_traces = []
    pre = 20  # frames before rotation onset
    post = 70  # frames after rotation onset

    for segment in rotation_segments:
        onset = segment[0]
        start = onset - pre
        end = onset + post

        line_trace = df_over_f_line[start:end]
        frame_trace = df_over_f_frame[start:end]
        line_traces.append(line_trace)
        frame_traces.append(frame_trace)

        axis.plot(
            line_trace,
            color=color_line_based,
            label="Line based derotation",
            linewidth=0.5,
            alpha=0.2,
        )
        axis.plot(
            frame_trace,
            color=color_frame_based,
            label="Frame based derotation",
            linewidth=0.5,
            alpha=0.2,
        )

    mean_line_trace = np.mean(line_traces, axis=0)
    mean_frame_trace = np.mean(frame_traces, axis=0)

    axis.plot(
        mean_line_trace,
        color=color_line_based,
        label="Line based derotation",
        linewidth=1.5,
    )
    axis.plot(
        mean_frame_trace,
        color=color_frame_based,
        label="Frame based derotation",
        linewidth=1.5,
    )
    axis.hlines(
        y=0.6,
        xmin=pre,
        xmax=rotation_segments[0][-1] - rotation_segments[0][0] + pre,
        color=rotation_on_color,
        linewidth=4,
        alpha=1,
    )

    axis.set_ylim(-1.5, 1)

    axis.set_ylabel("Mean ΔF/F$_0$")
    axis.set_title(
        f"{'Clockwise' if direction == 1.0 else 'Counter-clockwise'} "
        f"at {speed} deg/s"
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

mean_plot_CCW.yaxis.set_visible(False)
mean_plot_CCW.spines["left"].set_visible(False)

mean_plot_CCW.set_xlabel("Frames")
mean_plot_CW.set_xlabel("Frames")

# customize ticks and spines
line_plot.spines["left"].set_bounds(-1, 3)
line_plot.spines["bottom"].set_bounds(0, number_of_frames_to_plot)
line_plot.yaxis.set_major_locator(MultipleLocator(1))
line_plot.set_yticks(np.arange(-1, 4.0, 1.0))
mean_plot_CW.spines["bottom"].set_bounds(0, pre + post)
mean_plot_CCW.spines["bottom"].set_bounds(0, pre + post)

# legend
ax4 = plt.subplot2grid((10, 10), (0, 9), rowspan=1, colspan=1)
ax4.text(
    1.7,
    2.3,
    "Line based derotation",
    color=color_line_based,
    ha="right",
    fontweight="bold",
)
ax4.text(
    1.7,
    1.6,
    "Frame based derotation",
    color=color_frame_based,
    ha="right",
    fontweight="bold",
)
ax4.text(
    1.7,
    0.9,
    "Rotation on",
    color=rotation_on_color,
    ha="right",
    fontweight="bold",
)
ax4.axis("off")

# customize subplot positions to reduce gaps
plt.subplots_adjust(wspace=2, hspace=0.5)

gap_target = 0.01
pos_cw = mean_plot_CW.get_position()
pos_ccw = mean_plot_CCW.get_position()
current_gap = pos_ccw.x0 - pos_cw.x1
if current_gap > gap_target:
    shift = (current_gap - gap_target) / 2.0
    mean_plot_CW.set_position(
        [pos_cw.x0, pos_cw.y0, pos_cw.width + shift, pos_cw.height]
    )
    mean_plot_CCW.set_position(
        [pos_ccw.x0 - shift, pos_ccw.y0, pos_ccw.width + shift, pos_ccw.height]
    )

#  save the figure
plt.savefig("figure_3.png", dpi=300, bbox_inches="tight", pad_inches=0.5)
plt.savefig("figure_3.svg", bbox_inches="tight", pad_inches=0.5)
plt.close()
