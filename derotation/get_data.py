import numpy as np
import tifffile as tiff
from scipy.io import loadmat

from derotation.read_binary import read_rc2_bin


def get_paths(path):
    path_tif = path / "imaging/runtest_00001.tif"
    path_aux = path / "aux_stim/202303271657_21_005.bin"
    path_config = "derotation/config.yml"
    path_randperm = path / "stimlus_randperm.mat"

    return path_tif, path_aux, path_config, path_randperm


def get_data(path):
    path_tif, path_aux, path_config, path_randperm = get_paths(path)

    image = tiff.imread(path_tif)

    pseudo_random = loadmat(path_randperm)
    full_rotation_blocks_direction = pseudo_random["stimulus_random"][:, 2] > 0
    direction = np.ones(5)
    direction[full_rotation_blocks_direction[0:5]] = -1

    data, dt, chan_names, config = read_rc2_bin(path_aux, path_config)
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
