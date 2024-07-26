from pathlib import Path

import numpy as np
import pynapple as nap
from skimage.measure import find_contours

from derotation.postprocessing.neuropil_subtraction import neuropil_subtraction


def load_suite2p_data(path: Path) -> nap.NWBFile:
    suite2p_data = nap.load_file(path)
    return suite2p_data


def get_dff(suite2p_data: nap.NWBFile) -> tuple:
    raw_fluorescence = suite2p_data["RoiResponseSeries"]

    neuropil = suite2p_data["Neuropil"]

    dff, r = neuropil_subtraction(
        raw_fluorescence[:].values.T, neuropil[:].values.T
    )

    dff = nap.Tsd(t=raw_fluorescence.t, d=dff.T)
    timebase = raw_fluorescence.t

    return dff, timebase


def get_plane_segmentation(suite2p_data: nap.NWBFile) -> tuple:
    plane_seg = (
        suite2p_data.nwb.processing["ophys"]
        .data_interfaces["ImageSegmentation"]
        .plane_segmentations["PlaneSegmentation"]
    )

    ROI_centroids = plane_seg.ROICentroids[:]
    is_cell = plane_seg.Accepted[:]
    labels = plane_seg.image_mask[:]

    contours = [find_contours(c)[0] for c in labels[is_cell.astype(bool)]]

    return ROI_centroids, is_cell, labels, contours


def load_registered_binary(path: Path, shape: tuple) -> np.memmap:
    registered = np.memmap(path, shape=shape, dtype="int16")
    return registered
