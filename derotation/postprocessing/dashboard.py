from pathlib import Path

import fastplotlib as fpl
import numpy as np
import pandas as pd
from ipywidgets import AppLayout

from derotation.postprocessing.load_processed_data import (
    get_dff,
    get_plane_segmentation,
    load_registered_binary,
    load_suite2p_data,
)

# warnings.filterwarnings("ignore", category=DeprecationWarning)

# ⚁⚀ ⚯ ⚁⚀ ⚯ ⚁⚀ ⚯ ⚁⚀ ⚯ ⚁⚀ ⚯ ⚁⚀ ⚯ ⚁⚀ ⚯ ⚁⚀ ⚯ ⚁⚀ ⚯ ⚁⚀ ⚯
# Load data

path_to_nwb = Path(
    "/Users/lauraporta/Source/local/NWB_conversions/data/suite2p.nwb"
)

suite2p_data = load_suite2p_data(path_to_nwb)

dff, timebase = get_dff(suite2p_data)

ROI_centroids, is_cell, labels, contours = get_plane_segmentation(suite2p_data)


# Load csv file containing rotation information
rotation_info = pd.read_csv("rotation_info.csv")


# import the registered binary as it is not included in the NWB file
path_to_bin_file = Path(
    "/Users/laura/data/derotation/raw/230802_CAA_1120182/derotated/archive/test/suite2p/plane0/data.bin"
)

registered = load_registered_binary(path_to_bin_file, (512, 512, len(dff)))

# ⚁⚀ ⚯ ⚁⚀ ⚯ ⚁⚀ ⚯ ⚁⚀ ⚯ ⚁⚀ ⚯ ⚁⚀ ⚯ ⚁⚀ ⚯ ⚁⚀ ⚯ ⚁⚀ ⚯ ⚁⚀ ⚯
# Starting point of the dashboard


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


def dashboard():
    AppLayout(
        header=rotation_fig.show(maintain_aspect=False),
        left_sidebar=iw.show(),
        right_sidebar=fig_single_trace.show(maintain_aspect=False),
    )


if __name__ == "__main__":
    dashboard()
