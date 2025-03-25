"""
This module contains the function derotate, which is used to run the full
derotation pipeline on a single dataset. It can be called externally from the
command line or from a script to run multiple datasets in parallel.
Configurations are generated based on the provided dataset folder and output
folder.
"""

import logging
import traceback
from pathlib import Path

from derotation.analysis.full_derotation_pipeline import FullPipeline
from derotation.config.load_config import load_config, update_config_paths


def derotate(dataset_folder: Path, output_folder: str) -> float:
    """
    Run the full derotation pipeline on a single dataset.

    Parameters
    ----------
    dataset_folder : Path
        The path to the dataset folder.
    output_folder : str
        The path to the output folder in which to save the results.

    Returns
    -------
    float
        The metric calculated by the pipeline.

    Raises
    ------
    Exception
        If the pipeline fails.
    """
    # FULL DEROTATION PIPELINE
    # find tif and bin files
    bin_path = list(dataset_folder.rglob("*rotation_*001.bin"))[0]
    tif_path = list(dataset_folder.rglob("rotation_00001.tif"))[0]

    # Load the config template and update paths
    config = load_config(pipeline="full")
    config = update_config_paths(
        config, tif_path, bin_path, dataset_folder, output_folder, kind="full"
    )

    logging.info("Running full derotation pipeline")

    # Run the pipeline
    try:
        derotator = FullPipeline(config)
        derotator()
        return derotator.metric
    except Exception as e:
        logging.error("Full derotation pipeline failed")
        logging.error(e.args)
        logging.error(traceback.format_exc())
        raise e
