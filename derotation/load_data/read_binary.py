import numpy as np


def read_rc2_bin(path_aux, config):
    # Number of channels
    chan_names = config["channel_names"]
    n_channels = len(chan_names)

    # Read binary data saved by rc2 (int16)
    # Channels along the columns, samples along the rows.
    data = np.fromfile(path_aux, dtype=np.int16)
    data = data.reshape((-1, n_channels))

    # Use config file to determine offset and scale of each channel
    offsets = config["offset"]
    scales = config["scale"]

    # Transform the data.
    # Convert 16-bit integer to volts between -10 and 10.
    data = -10 + 20 * (data + 2**15) / 2**16
    # Use offset and scale to transform to correct units (cm/s etc.)
    data = (data - offsets) / scales
    # Also return sampling period
    dt = config["rotation_increment"]

    return data, dt, chan_names, config
