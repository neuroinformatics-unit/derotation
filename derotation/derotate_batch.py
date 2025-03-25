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

import yaml

from derotation.analysis.full_derotation_pipeline import FullPipeline


def update_config_paths(
    config: dict,
    tif_path: Path,
    bin_path: Path,
    dataset_path: Path,
    output_folder: str,
    kind: str = "full",
) -> dict:
    """
    Update the paths in the config dictionary based on the provided arguments.

    Parameters
    ----------
    config : dict
        The configuration dictionary.
    tif_path : Path
        The path to the tif file to be derotated.
    bin_path : Path
        The path to the bin file containing analog signals.
    dataset_path : Path
        The path to the dataset folder.
    output_folder : str
        The path to the output folder in which to save the results.
    kind : str, optional
        Which derotation pipeline to run, by default "full".

    Returns
    -------
    dict
        The updated configuration dictionary.
    """
    # Set config paths based on provided arguments
    config["paths_read"]["path_to_randperm"] = str(
        Path(dataset_path).parent / "stimlus_randperm.mat"
    )
    config["paths_read"]["path_to_aux"] = str(bin_path)
    config["paths_read"]["path_to_tif"] = str(tif_path)

    # Set output paths to the specified output_folder
    config["paths_write"]["debug_plots_folder"] = str(
        Path(output_folder) / "derotation" / f"debug_plots_{kind}"
    )
    config["paths_write"]["logs_folder"] = str(
        Path(output_folder) / "derotation" / "logs"
    )
    config["paths_write"]["derotated_tiff_folder"] = str(
        Path(output_folder) / "derotation/"
    )
    config["paths_write"]["saving_name"] = f"derotated_{kind}"

    return config


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
    this_module_path = Path(__file__).parent

    # FULL DEROTATION PIPELINE
    # find tif and bin files
    bin_path = list(dataset_folder.rglob("*rotation_*001.bin"))[0]
    tif_path = list(dataset_folder.rglob("rotation_00001.tif"))[0]

    # Load the config template and update paths

    config_template_path = this_module_path / Path("config/full_rotation.yml")
    with open(config_template_path, "r") as f:
        config = yaml.safe_load(f)

    config = update_config_paths(
        config, tif_path, bin_path, dataset_folder, output_folder, kind="full"
    )

    # Create output directories if they don't exist
    Path(config["paths_write"]["debug_plots_folder"]).mkdir(
        parents=True, exist_ok=True
    )
    Path(config["paths_write"]["logs_folder"]).mkdir(
        parents=True, exist_ok=True
    )
    Path(config["paths_write"]["derotated_tiff_folder"]).mkdir(
        parents=True, exist_ok=True
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
