from pathlib import Path

import numpy as np
import tifffile as tiff
import yaml
from scipy.io import loadmat

from derotation.load_data.read_binary import read_rc2_bin


def get_paths():
    path_config = "derotation/config.yml"

    with open(path_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    root = Path(config["paths"]["root"])
    path_to_randperm = root / Path(config["paths"]["path-to-randperm"])
    path_to_dataset_folder = root / config["paths"]["dataset-folder"]
    path_to_aux = path_to_dataset_folder / Path(config["paths"]["path-to-aux"])
    path_to_tif = path_to_dataset_folder / Path(config["paths"]["path-to-tif"])
    filename = config["paths"]["path-to-tif"]

    return (
        path_to_tif,
        path_to_aux,
        config,
        path_to_randperm,
        path_to_dataset_folder,
        filename,
    )


def get_data():
    (
        path_tif,
        path_aux,
        config,
        path_randperm,
        path_to_dataset_folder,
        filename,
    ) = get_paths()

    image = tiff.imread(path_tif)

    pseudo_random = loadmat(path_randperm)
    full_rotation_blocks_direction = pseudo_random["stimulus_random"][:, 2] > 0
    direction = np.where(
        full_rotation_blocks_direction, -1, 1
    )  # 1 is counterclockwise, -1 is clockwise

    data, dt, chan_names, config = read_rc2_bin(path_aux, config)
    data_dict = {chan: data[:, i] for i, chan in enumerate(chan_names)}

    frame_clock = data_dict["scanimage_frameclock"]
    line_clock = data_dict["scanimage_lineclock"]
    full_rotation = data_dict["PI_rotON"]
    rotation_ticks = data_dict["PI_rotticks"]

    return (
        image,
        frame_clock,
        line_clock,
        full_rotation,
        rotation_ticks,
        dt,
        config,
        direction,
        path_to_dataset_folder,
        filename,
    )
