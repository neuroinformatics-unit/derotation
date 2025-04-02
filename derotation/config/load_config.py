from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(pipeline: str = "full") -> Dict[str, Any]:
    """ "
    Load the configuration file for the pipeline.

    Parameters
    ----------
    pipeline : str
        The pipeline to use. Options are 'full' or 'incremental'.
    """
    this_module_path = Path(__file__).parent
    if pipeline == "full":
        with open(this_module_path / "full_rotation.yml", "r") as f:
            config = yaml.safe_load(f)
    elif pipeline == "incremental":
        with open(this_module_path / "incremental_rotation.yml", "r") as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("Pipeline must be 'full' or 'partial'")
    return config


def update_config_paths(
    config: dict,
    tif_path: Path,
    aux_path: Path,
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
    aux_path : Path
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
    config["paths_read"]["path_to_aux"] = str(aux_path)
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
