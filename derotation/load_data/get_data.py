from pathlib import Path

import numpy as np
import tifffile as tiff
import yaml
from scipy.io import loadmat

from derotation.load_data.read_binary import read_rc2_bin


def get_paths(
    dataset_name="pollen",
):
    path_config = "derotation/config.yml"

    with open(path_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    path = Path(config["path"])

    if dataset_name == "pollen":
        path = path / "230327_pollen"
        path_tif = path / "imaging/runtest_00001.tif"
        path_aux = path / "aux_stim/202303271657_21_005.bin"
        path_randperm = path / "stimlus_randperm.mat"
    elif dataset_name == "far_focus_cells":
        path_randperm = path / "230327_pollen" / "stimlus_randperm.mat"
        path = path / "AK_1119329_hR_RSPd_mid_rotation"
        path_tif = path / "rotation_00001.tif"
        path_aux = path / "aux_files/rotation/rotation_1_001.bin"
    elif dataset_name == "grid":
        path_randperm = path / "230327_pollen" / "stimlus_randperm.mat"
        path = path / "230731_25_micron_grid"
        path_tif = (
            path / "imaging" / "rotation_zf2_all_speeds_00002_enhanced.tif"
        )
        path_aux = path / "aux_stim" / "230731_grid_1_001.bin"

    return path_tif, path_aux, config, path_randperm


def get_data():
    path_tif, path_aux, config, path_randperm = get_paths(dataset_name="grid")

    image = tiff.imread(path_tif)

    pseudo_random = loadmat(path_randperm)
    full_rotation_blocks_direction = pseudo_random["stimulus_random"][:, 2] > 0
    direction = np.where(
        full_rotation_blocks_direction, -1, 1
    )  # 1 is counterclockwise, -1 is clockwise

    data, dt, chan_names, config = read_rc2_bin(path_aux, config)
    data_dict = {chan: data[:, i] for i, chan in enumerate(chan_names)}

    frame_clock = data_dict["scanimage_frameclock"]
    line_clock = data_dict["camera"]
    full_rotation = data_dict["PI_rotCW"]
    rotation_ticks = data_dict["Vistim_ttl"]

    return (
        image,
        frame_clock,
        line_clock,
        full_rotation,
        rotation_ticks,
        dt,
        config,
        direction,
    )
