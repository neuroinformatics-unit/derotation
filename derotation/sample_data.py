"""Fetch and load sample datasets for derotation examples.

This module provides functions for fetching sample data used in examples.
The data are stored in a remote repository and are downloaded to the user's
local machine the first time they are used.
"""

from pathlib import Path

import pooch

# URL to the remote data repository on GIN
DATA_URL = "https://gin.g-node.org/l.porta/derotation_examples/src/master/"

# Save data in ~/.derotation/data
DATA_DIR = Path("~", ".derotation", "data").expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# File registry with SHA-256 checksums
REGISTRY = {
    "angles_per_line.npy": (  # noqa: E501
        "sha256:0a5dab081acdfb47ebd5cdda4094cc622b1ff708c63f6fadc1e7898d30789896"
    ),
    "rotation_sample.tif": (  # noqa: E501
        "sha256:ad8aae61cda9495d9006fb37a1f87be8b6dd6477318b1e8a409417cace208f56"
    ),
    "analog_signals.npy": (  # noqa: E501
        "sha256:e04437f86d07db8077c256f912e2bcca75794712d66715f301b99cd9a8d66d95"
    ),
    "stimulus_randperm.csv": (  # noqa: E501
        "sha256:214617992d786ee210a7d2f22dbe7420997016846a79c673979913a5b77f0a11"
    ),
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
            f"File '{filename}' not found in registry."
            f"Available files: {list(REGISTRY.keys())}"
        )

    return Path(SAMPLE_DATA.fetch(filename, progressbar=True))


def get_data_path(filename: str) -> Path:
    """Get the path to a sample data file, downloading it if necessary.

    Parameters
    ----------
    filename : str
        Name of the file to get.

    Returns
    -------
    Path
        Path to the data file.
    """
    return fetch_data(filename)
