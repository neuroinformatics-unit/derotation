"""Fetch and load sample datasets for derotation examples.

This module provides functions for fetching sample data used in examples.
The data are stored in a remote repository and are downloaded to the user's
local machine the first time they are used.
"""

from pathlib import Path

import pooch

# URL to the remote data repository on GIN
DATA_URL = "https://gin.g-node.org/l.porta/derotation_examples/raw/master/"

# Save data in ~/.derotation/data
DATA_DIR = Path("~", ".derotation", "data").expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# File registry without checksums (just for file validation)
REGISTRY = {
    "angles_per_line.npy": None,
    "rotation_sample.tif": None,
    "analog_signals.npy": None,
    "stimulus_randperm.csv": None,
}

# Create a download manager
SAMPLE_DATA = pooch.create(
    path=DATA_DIR,
    base_url=DATA_URL,
    registry=REGISTRY,
)


def fetch_data(filename: str) -> Path:
    """Fetch a sample data file.

    Parameters
    ----------
    filename : str
        Name of the file to fetch.

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    if filename not in REGISTRY:
        raise ValueError(
            f"File '{filename}' not found in registry. "
            f"Available files: {list(REGISTRY.keys())}"
        )

    return Path(SAMPLE_DATA.fetch(filename, progressbar=True))
