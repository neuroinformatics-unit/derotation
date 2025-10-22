"""Fetch and load sample datasets for derotation examples.

This module provides functions for fetching sample data used in examples.
The data are stored in a remote repository and are downloaded to the user's
local machine the first time they are used.
"""

from pathlib import Path

import pooch

# URL to the remote data repository on GIN
DATA_URL = (
    "https://gin.g-node.org/neuroinformatics/derotation_examples/raw/master/"
)

# Save data in ~/.derotation/data
DATA_DIR = Path("~", ".derotation", "data").expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# File registry with SHA256 checksums for data integrity validation
REGISTRY = {
    "angles_per_line.npy": (
        "0a5dab081acdfb47ebd5cdda4094cc622b1ff708c63f6fadc1e7898d30789896"
    ),
    "rotation_sample.tif": (
        "ad8aae61cda9495d9006fb37a1f87be8b6dd6477318b1e8a409417cace208f56"
    ),
    "analog_signals.npy": (
        "e04437f86d07db8077c256f912e2bcca75794712d66715f301b99cd9a8d66d95"
    ),
    "stimulus_randperm.csv": (
        "214617992d786ee210a7d2f22dbe7420997016846a79c673979913a5b77f0a11"
    ),
    "angles_per_frame.npy": (
        "5ff7cf196ffa9714a8ef7124c70df4fc8b5085ee092198f16019b358103f5fb6"
    ),
    "figure3/analog_signals.bin": (
        "1bcf9c9f76020873214b3fcf3263b63c172c1bd0496df555eec57f17866293d9"
    ),
    "figure3/rotated_stack.tif": (
        "ac71e3a8bec65949b640104bdd3a3678cd663570a5b29352d1eef83d9ab3dc1e"
    ),
    "figure3/stimlus_random_permutations.mat": (
        "7bd5f19996073918f0528588afec305148ad999fa2d83b530d19a2fe17234809"
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
            f"File '{filename}' not found in registry. "
            f"Available files: {list(REGISTRY.keys())}"
        )

    return Path(SAMPLE_DATA.fetch(filename, progressbar=True))
