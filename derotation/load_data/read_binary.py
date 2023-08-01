import numpy as np


def read_rc2_bin(path_aux, config):
    # Number of channels
    # chan_names = config["nidaq"]["ai"]["channel_names"]
    chan_names = [
        "camera",
        "scanimage_frameclock",
        "scanimage_lineclock",
        "photodiode2",
        "PI_rotON",
        "PI_rotticks",
    ]
    n_channels = len(chan_names)

    # Read binary data saved by rc2 (int16)
    # Channels along the columns, samples along the rows.
    data = np.fromfile(path_aux, dtype=np.int16)
    data = data.reshape((-1, n_channels))

    # Use config file to determine offset and scale of each channel
    offsets = config["nidaq"]["ai"]["offset"]
    scales = config["nidaq"]["ai"]["scale"]

    # Transform the data.
    # Convert 16-bit integer to volts between -10 and 10.
    data = -10 + 20 * (data + 2**15) / 2**16
    # Use offset and scale to transform to correct units (cm/s etc.)
    data = (data - offsets) / scales
    # Also return sampling period
    dt = 1 / config["nidaq"]["ai"]["rate"]

    return data, dt, chan_names, config
