import warnings
from pathlib import Path

import allensdk.brain_observatory.dff as dff_module
import fastplotlib as fpl
import numpy as np
import pandas as pd
import pynapple as nap
from allensdk.brain_observatory.r_neuropil import NeuropilSubtract
from ipywidgets import AppLayout
from skimage.measure import find_contours

warnings.filterwarnings("ignore", category=DeprecationWarning)

# load data

suite2p_data = nap.load_file("suite2p.nwb")
rotation_info = pd.read_csv("rotation_info.csv")


# Isolate first main objects
# fluorescence
raw_fluorescence = suite2p_data["RoiResponseSeries"]

# neuropil
neuropil = suite2p_data["Neuropil"]

# rotation_angles
rotation_angles = nap.Tsd(
    t=raw_fluorescence.t, d=rotation_info["rotation_angle"]
)

# when is the rotation happening: interval set
rotation_stim = nap.TsdFrame(
    t=raw_fluorescence.t,
    d=rotation_info[["direction", "speed", "rotation_count"]].values,
    columns=["direction", "speed", "rotation_count"],
)

rotation_on_intervals = rotation_angles.threshold(0).time_support.union(
    rotation_angles.threshold(0, "below").time_support
)

stimuli_start_info = nap.TsdFrame(
    rotation_stim.restrict(rotation_on_intervals)
    .as_dataframe()
    .drop_duplicates("rotation_count")
    .dropna()
)


def neuropil_subtraction(f, f_neu):
    #  use default parameters for all methods
    neuropil_subtraction = NeuropilSubtract()
    neuropil_subtraction.set_F(f, f_neu)
    neuropil_subtraction.fit()

    r = neuropil_subtraction.r

    f_corr = f - r * f_neu

    dff = 100 * dff_module.compute_dff_windowed_median(
        f_corr, median_kernel_long=1213, median_kernel_short=23
    )

    return dff, r


dff, r = neuropil_subtraction(
    raw_fluorescence[:].values.T, neuropil[:].values.T
)

dff = nap.dff(t=raw_fluorescence.t, d=dff.T)

stimuli_start_info_df = stimuli_start_info.as_dataframe()
rotation_on_intervals_df = rotation_on_intervals.as_dataframe()
for key in stimuli_start_info_df.keys()[:2]:
    for item in stimuli_start_info_df[key].unique():
        subset = stimuli_start_info_df[stimuli_start_info_df[key] == item]
        starts = subset.index
        rotation_on_intervals_df[rotation_on_intervals_df["start"] == starts]

# where is the data in nwb file?
plane_seg = (
    suite2p_data.nwb.processing["ophys"]
    .data_interfaces["ImageSegmentation"]
    .plane_segmentations["PlaneSegmentation"]
)

ROI_centroids = plane_seg.ROICentroids[:]
is_cell = plane_seg.Accepted[:]
labels = plane_seg.image_mask[:]


contours = [find_contours(c)[0] for c in labels[is_cell.astype(bool)]]


# import the registered binary
path_to_bin_file = Path(
    "/Users/laura/data/derotation/raw/230802_CAA_1120182/derotated/archive/test/suite2p/plane0/data.bin"
)

shape_image = (16516, 256, 256)
registered = np.memmap(path_to_bin_file, shape=shape_image, dtype="int16")


def find_ROI(nearest):
    x, y, _ = np.asarray(nearest.data.value).mean(axis=0)
    # print(f'x:{x}, y:{y}')
    closest_x = np.abs(ROI_centroids[:, 0] - x).argmin()
    closest_y = np.abs(ROI_centroids[:, 1] - y).argmin()
    # print(f'closest x: {closest_x}, closest_y: {closest_y}')
    if closest_x != closest_y:
        delta_x = ROI_centroids[closest_x, 0] - x
        delta_y = ROI_centroids[closest_y, 1] - y
        if delta_x < delta_y:
            return closest_x
        else:
            return closest_y
    else:
        return closest_x


# Panel 1
colors = {key: "white" for key in range(len(rotation_info["rotation_angle"]))}
for i, row in rotation_info.iterrows():
    if row["direction"] == -1:
        colors[i] = "green"
    if row["direction"] == 1:
        colors[i] = "magenta"

rotation_fig = fpl.Figure((3, 1), size=(1200, 300))
angles_top_plot = rotation_fig[0, 0].add_line(
    data=rotation_info["rotation_angle"],
    thickness=1,
    colors=list(colors.values()),
)

region_selector = angles_top_plot.add_linear_region_selector()
zoomed_init = region_selector.get_selected_data()
zoomed_x = rotation_fig[1, 0].add_line(zoomed_init)

selector = angles_top_plot.add_linear_selector()

selection_boundaries = region_selector.get_selected_indices()
sliced_dff = dff[selection_boundaries[0] : selection_boundaries[-1]]

dff_lines = rotation_fig[2, 0].add_line_stack(
    np.asarray(sliced_dff).T[is_cell],
    cmap="jet",
    thickness=1,
    separation=1,
)


@region_selector.add_event_handler("selection")
def slice_dff_array(ev):
    global zoomed_x
    selected_data = ev.get_selected_data()
    rotation_fig[1, 0].remove_graphic(zoomed_x)
    zoomed_x = rotation_fig[1, 0].add_line(selected_data)
    rotation_fig[1, 0].auto_scale()

    global dff_lines
    global sliced_dff
    selection_boundaries = ev.get_selected_indices()
    sliced_dff = dff[selection_boundaries[0] : selection_boundaries[-1]]

    rotation_fig[2, 0].clear()
    dff_lines = rotation_fig[2, 0].add_line_stack(
        np.asarray(sliced_dff).T[is_cell],
        cmap="jet",
        thickness=1,
        separation=1,
    )
    rotation_fig[2, 0].auto_scale()

    fig_single_trace[0, 0].clear()
    fig_single_trace[0, 0].add_line(
        data=np.asarray(sliced_dff).T[idx],
        thickness=2,
        cmap="plasma",
    )
    fig_single_trace[0, 0].auto_scale()


# Panel 2
iw = fpl.ImageWidget(
    registered[: len(rotation_info["rotation_angle"])], cmap="gnuplot2"
)
iw.show()
contours_graphic = iw.figure[0, 0].add_line_collection(
    contours, thickness=2, colors="green"
)

# Panel 3
fig_single_trace = fpl.Figure()
idx = 0
single_line = fig_single_trace[0, 0].add_line(
    data=np.asarray(sliced_dff).T[idx],
    thickness=2,
    cmap="plasma",
)

rotation_fig.show(maintain_aspect=False)


def set_selected_component(ev):
    xy = iw.figure[0, 0].map_screen_to_world(ev)[:-1]
    global nearest
    nearest = fpl.utils.get_nearest_graphics(xy, contours_graphic)[0]
    contours_graphic.colors = "green"
    nearest.colors = "w"

    global idx
    global single_line
    idx = find_ROI(nearest)
    fig_single_trace[0, 0].clear()
    single_line = fig_single_trace[0, 0].add_line(
        data=np.asarray(sliced_dff).T[idx],
        thickness=2,
        cmap="plasma",
    )
    fig_single_trace[0, 0].auto_scale()


iw.figure.renderer.add_event_handler(set_selected_component, "click")
selector.add_ipywidget_handler(iw.sliders["t"], step=1)

# Handling ipywidgets layout
AppLayout(
    header=rotation_fig.show(maintain_aspect=False),
    left_sidebar=iw.show(),
    right_sidebar=fig_single_trace.show(maintain_aspect=False),
)
