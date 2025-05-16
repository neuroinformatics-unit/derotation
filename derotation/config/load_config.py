from pathlib import Path
from typing import Any, Dict

import yaml


def load_config() -> Dict[str, Any]:
    """
    Load a pre-filled configuration file for the derotation pipeline.

    Returns
    -------
    dict
        The configuration dictionary.
    """
    this_module_path = Path(__file__).parent
    with open(this_module_path / "config_template.yml", "r") as f:
        config = yaml.safe_load(f)

    return config


def update_config_paths(
    config: dict,
    tif_path: str,
    aux_path: str,
    stim_randperm_path: str,
    output_folder: str,
    folder_suffix: str = "",
) -> dict:
    """
    Update the paths in the config dictionary based on the provided arguments.

    Parameters
    ----------
    config : dict
        The configuration dictionary.
    tif_path : str
        The path to the tif file to be derotated.
    aux_path : str
        The path to the bin file containing analog signals.
    dataset_path : str
        The path to the dataset folder.
    output_folder : str
        The path to the output folder in which to save the results.
    folder_suffix : str, optional
        A suffix to append to the output folder names (default is an empty
        string).

    Returns
    -------
    dict
        The updated configuration dictionary.
    """
    # Set config paths based on provided arguments
    config["paths_read"]["path_to_randperm"] = stim_randperm_path
    config["paths_read"]["path_to_aux"] = aux_path
    config["paths_read"]["path_to_tif"] = tif_path

    # Set output paths to the specified output_folder
    config["paths_write"]["debug_plots_folder"] = str(
        Path(output_folder)
        / "derotation"
        / f"debug_plots{f'_{folder_suffix}' if folder_suffix else ''}"
    )
    config["paths_write"]["logs_folder"] = str(
        Path(output_folder) / "derotation" / "logs"
    )
    config["paths_write"]["derotated_tiff_folder"] = str(Path(output_folder))
    config["paths_write"]["saving_name"] = (
        f"derotated{f'_{folder_suffix}' if folder_suffix else ''}"
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

    return config
