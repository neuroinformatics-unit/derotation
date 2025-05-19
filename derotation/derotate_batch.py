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
from derotation.analysis.incremental_derotation_pipeline import (
    IncrementalPipeline,
)
from derotation.config.load_config import load_config, update_config_paths


def derotate(
    dataset_folder: Path,
    output_folder: str,
    path_to_stimulus_randperm: str,
    glob_naming_pattern_tif: str,
    glob_naming_pattern_bin: str,
    folder_suffix: str = "full",
):
    """
    Run the derotation pipeline on a single dataset. This function is
    responsible for loading the configuration, updating the paths, the
    pipeline choice, and running the pipeline.

    Parameters
    ----------
    dataset_folder : Path
        The path to the dataset folder.
    output_folder : str
        The path to the output folder in which to save the results.
    path_to_stimulus_randperm : str
        The path to the stimulus random permutation file.
    glob_naming_pattern_tif : str
        The glob naming pattern for the tif file.
    glob_naming_pattern_bin : str
        The glob naming pattern for the bin file.
    folder_suffix : str, optional
        The suffix to append to the output folder name, by default "full".
        This is used to differentiate between full and incremental pipelines.

    Raises
    ------
    Exception
        If the pipeline fails.
    """
    # find tif and bin files
    bin_path = list(dataset_folder.rglob(glob_naming_pattern_bin))[0]
    tif_path = list(dataset_folder.rglob(glob_naming_pattern_tif))[0]

    # Load the config template and update paths
    config = load_config()
    config = update_config_paths(
        config=config,
        tif_path=str(tif_path),
        aux_path=str(bin_path),
        stim_randperm_path=str(path_to_stimulus_randperm),
        output_folder=output_folder,
        folder_suffix=folder_suffix,
    )

    logging.info(f"Running {folder_suffix} derotation pipeline")

    # Run the pipeline
    try:
        if folder_suffix == "full":
            derotator = FullPipeline(config)
        else:
            derotator = IncrementalPipeline(config)
        derotator()
    except Exception as e:
        logging.error("Derotation pipeline failed")
        logging.error(e.args)
        logging.error(traceback.format_exc())
        raise e
